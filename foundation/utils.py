from iris.foundation.exception import TaskError
from iris.foundation.types import ProblemType
from iris.foundation.exception import DataLoadingError 
from iris.foundation.types import LoadMode
from typing import Union, Iterator
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import requests
import os
import io
import csv
import chardet

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def determine_task_type(y: pd.Series, is_timeseries: bool = False) -> ProblemType:
    """
    Intelligently infers the machine learning task type from the target variable.
    """
    try:
        if is_timeseries:
            return ProblemType.TIME_SERIES_FORECASTING

        y_clean = y.dropna()
        if y_clean.empty:
            raise ValueError("Target variable is empty or contains only NaNs.")

        unique_count = y_clean.nunique()

        if pd.api.types.is_bool_dtype(y_clean):
            return ProblemType.BINARY_CLASSIFICATION

        if pd.api.types.is_float_dtype(y_clean):
            # Check for "fake floats" (e.g., 1.0, 2.0)
            is_real_float = not np.all(np.mod(y_clean, 1) == 0)
            if is_real_float:
                return ProblemType.REGRESSION

        if pd.api.types.is_object_dtype(y_clean) or isinstance(y_clean.dtype, pd.CategoricalDtype):
            if unique_count <= 2:
                return ProblemType.BINARY_CLASSIFICATION
            return ProblemType.MULTICLASS_CLASSIFICATION

        CLASSIFICATION_THRESHOLD = 20
        
        if unique_count <= 2:
            return ProblemType.BINARY_CLASSIFICATION
        elif unique_count <= CLASSIFICATION_THRESHOLD:
            return ProblemType.MULTICLASS_CLASSIFICATION
        else:
            return ProblemType.REGRESSION

    except Exception as e:
        raise TaskError(f"Failed to infer task type: {str(e)}")

def load_dataset(src: Union[str, pd.DataFrame],
                 mode: LoadMode = LoadMode.Default,
                 chunk_size: int = 10_000,
                 auto_clean: bool = True,
                 **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Smart Loader: Loads data from various sources into a Pandas DataFrame.
    Prioritizes Polars for high performance.
    """

    if isinstance(src, pd.DataFrame):
        if src.empty:
            raise DataLoadingError("Provided DataFrame is empty.")
        return _auto_fix_df(src) if auto_clean else src

    if not isinstance(src, str):
        raise DataLoadingError(f"Source must be a string or DataFrame, got {type(src)}")

    src = src.strip()
    
    # Try Polars First for Speed
    if pl is not None and mode != LoadMode.Lazy:
        try:
            df_pl = _load_with_polars(src, **kwargs)
            if df_pl is not None:
                return _polars_auto_fix(df_pl).to_pandas() if auto_clean else df_pl.to_pandas()
        except Exception:
            # Fallback to Pandas if Polars fails (e.g. complex CSV dialect, proprietary Excel)
            pass

    if _is_url(src):
        ext = _infer_extension(src)
        if mode == LoadMode.Lazy:
             raise DataLoadingError("Lazy mode currently supports only local files.")
        df = _load_from_url(src, ext, **kwargs)
    else:
        if not os.path.exists(src):
             raise DataLoadingError(f"File not found: {src}")
        ext = _infer_extension(src)
        if mode == LoadMode.Lazy:
            return _load_lazy(src, ext, chunk_size)
        df = _load_file(src, ext, **kwargs)

    if df is None or df.empty:
        raise DataLoadingError(f"Empty dataframe from: {src}")

    return _auto_fix_df(df) if auto_clean else df

def _load_with_polars(src: str, **kwargs) -> Union['pl.DataFrame', None]:
    """Attempts to load data using Polars."""
    try:
        if _is_url(src):
            # Polars doesn't support remote URLs directly in read_csv as robustly as pandas/requests yet for all cases,
            # but we can download to buffer.
            r = requests.get(src, timeout=30)
            r.raise_for_status()
            data = io.BytesIO(r.content)
            
            # infer type from url
            ext = _infer_extension(src)
            if ext == 'csv':
                # Map pandas 'sep' or 'delimiter' to polars 'separator'
                sep = kwargs.get('sep', kwargs.get('delimiter', ','))
                return pl.read_csv(data, separator=sep, null_values=['NA', 'nan', 'NaN', '?'])
            elif ext == 'parquet':
                return pl.read_parquet(data)
            elif ext == 'json':
                return pl.read_json(data)
            else:
                return None
        else:
            ext = _infer_extension(src)
            if ext == 'csv':
                sep = kwargs.get('sep', kwargs.get('delimiter', ','))
                return pl.read_csv(src, separator=sep, null_values=['NA', 'nan', 'NaN', '?'])
            elif ext == 'parquet':
                return pl.read_parquet(src)
            elif ext == 'json':
                return pl.read_json(src)
            return None
    except Exception:
        return None

def _polars_auto_fix(df: 'pl.DataFrame') -> 'pl.DataFrame':
    """High-performance cleaning using Polars Expressions."""
    try:
        # 1. Strip whitespace from all string columns
        # 2. Try to cast string columns to numbers (if they look like numbers)
        
        # We process string columns
        str_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        
        if not str_cols:
            return df.drop_nulls()

        # Expression to strip whitespace
        df = df.with_columns([
            pl.col(c).str.strip_chars() for c in str_cols
        ])
        
        # Expression to attempt numeric conversion
        # Polars 'cast' is strict, but we can use 'cast(pl.Float64, strict=False)' to get nulls on failure
        # Then check null count to decide whether to keep the cast
        
        ops = []
        for c in str_cols:
            # Check if column is effectively numeric
            # We do a quick check on a sample or just try cast
            # Using strict=False returns null for non-convertible
            # If > 80% are non-null after cast, we keep it.
            
            # Note: This is harder to do purely lazily without computing, 
            # so we might just do a simple "try cast" approach
            
            # Ideally we'd scan, but for now let's just do:
            pass 

        # Polars cleaning is complex to replicate 1:1 with the '80% numeric rule' efficiently without materializing.
        # For now, let's just return the stripped dataframe and let the downstream pandas logic handle complex type inference
        # OR we convert to pandas and let the existing robust _auto_fix_df handle the type mess,
        # but at least we got fast CSV parsing.
        
        return df.drop_nulls()
    except Exception:
        return df

def _is_url(path: str) -> bool:
    try:
        return urlparse(path).scheme in ("http", "https", "ftp")
    except:
        return False

def _infer_extension(path: str) -> str:
    path = urlparse(path).path
    return os.path.splitext(path)[1].lower().lstrip(".")

def _detect_encoding(path: str, n_bytes: int = 50_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(n_bytes)
        result = chardet.detect(data)
        return result["encoding"] or "utf-8"
    except Exception:
        return "utf-8"

def _detect_delimiter(src) -> str:
    try:
        sample = ""
        if isinstance(src, str):
             if os.path.exists(src):
                with open(src, "r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(2048)
        elif hasattr(src, 'read'):
            pos = src.tell()
            sample = src.read(2048)
            src.seek(pos)
        elif isinstance(src, io.StringIO):
             pos = src.tell()
             sample = src.read(2048)
             src.seek(pos)

        if not sample: 
            return ","

        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [",", ";", "|", "\t"]}
        return max(counts, key=counts.get, default=",")

def _load_file(path: str, ext: str, **kwargs) -> pd.DataFrame:
    try:
        if ext == "csv":
            enc = _detect_encoding(path)
            delim = _detect_delimiter(path)
            return pd.read_csv(path, encoding=enc, delimiter=delim, **kwargs)

        if ext in ("xls", "xlsx"):
            return pd.read_excel(path, **kwargs)

        if ext == "parquet":
            if pq is None:
                raise DataLoadingError("pyarrow required for Parquet")
            return pd.read_parquet(path)

        if ext == "json":
            return pd.read_json(path)

        try:
             enc = _detect_encoding(path)
             delim = _detect_delimiter(path)
             return pd.read_csv(path, encoding=enc, delimiter=delim, **kwargs)
        except:
             raise DataLoadingError(f"Unsupported file extension: .{ext}")

    except Exception as e:
        raise DataLoadingError(f"Error reading local file: {path}") from e

def _load_from_url(url: str, ext: str, **kwargs) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        b = io.BytesIO(r.content)
        t = io.StringIO(r.text)

        if ext == "csv":
            return pd.read_csv(t, delimiter=_detect_delimiter(t), **kwargs)

        if ext == "parquet":
            return pd.read_parquet(b)

        if ext in ("xls", "xlsx"):
            return pd.read_excel(b, **kwargs)

        if ext == "json":
            return pd.read_json(t)

        return pd.read_csv(t, delimiter=_detect_delimiter(t), **kwargs)

    except Exception as e:
        raise DataLoadingError(f"Error loading URL: {url}") from e

def _load_lazy(src: str, ext: str, chunk_size: int):
    if ext == "csv":
        return pd.read_csv(src, chunksize=chunk_size, delimiter=_detect_delimiter(src))
    if ext == "parquet":
        if pq is None:
            raise DataLoadingError("pyarrow required for Parquet")
        parquet_file = pq.ParquetFile(src)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
        return
    raise DataLoadingError(f"Lazy mode not supported for: .{ext}")

def _auto_fix_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df = df.drop_duplicates()
        df.columns = df.columns.astype(str).str.strip()

        for c in df.columns:
            if df[c].dtype == "object":
                try:
                    df[c] = df[c].str.strip()
                except Exception:
                    pass

                numeric = pd.to_numeric(df[c], errors="coerce")
                if numeric.notna().sum() > 0.8 * df[c].notna().sum():
                    df[c] = numeric
        return df
    except Exception:
        return df