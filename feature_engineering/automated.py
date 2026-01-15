import numpy as np
import pandas as pd
import polars as pl
from typing import List, Dict, Any, Optional
from iris.foundation.types import ProblemType
import logging

logger = logging.getLogger(__name__)

class AutoFeatureEngineer:
    """
    High-Performance Auto Feature Engineer using Polars.
    
    Strategies:
    1. Mathematical Transformations (Log, Sqrt) for skewed data.
    2. Arithmetic Interactions (A*B, A/B) for key numeric pairs.
    3. Grouped Aggregations (Mean/Std of Num by Cat) to capture context.
    4. Target Encoding for high-cardinality categoricals.
    """
    
    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.features_to_keep_: List[str] = []
        self.target_encodings_: Dict[str, Dict[str, float]] = {}
        self.global_mean_: float = 0.0
        self.skewed_cols_: List[str] = []
        self.interaction_pairs_: List[Tuple[str, str, str]] = [] # (col1, col2, op)
        self.agg_configs_: List[Tuple[str, str, str]] = [] # (group_col, target_col, stat)
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task: ProblemType) -> pd.DataFrame:
        # Convert to Polars
        try:
            df = pl.from_pandas(X)
            
            # Identify columns
            num_cols = [c for c, t in zip(df.columns, df.dtypes) if t in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            cat_cols = [c for c, t in zip(df.columns, df.dtypes) if t in [pl.Utf8, pl.Object, pl.Categorical]]
            
            # 1. Detect Skewed Columns for Log Transform
            if len(X) < 100_000: # Expensive on massive data
                stats = df.select([pl.col(c).skew() for c in num_cols]).to_dicts()[0]
                self.skewed_cols_ = [c for c, s in stats.items() if s is not None and abs(s) > 1.5]
            
            # 2. Setup Interactions (Top 10 numeric cols)
            # Ratios (A/B) and Products (A*B) are crucial for physical/chemical datasets
            top_num = num_cols[:10] 
            for i in range(len(top_num)):
                # Polynomial (Square)
                self.interaction_pairs_.append((top_num[i], None, '^2'))
                
                for j in range(i + 1, len(top_num)):
                    self.interaction_pairs_.append((top_num[i], top_num[j], '*'))
                    self.interaction_pairs_.append((top_num[i], top_num[j], '/'))
            
            # 3. Setup Grouped Aggregations
            for cat in cat_cols:
                n_unique = df[cat].n_unique()
                if 2 <= n_unique <= 50:
                    for num in num_cols[:3]: 
                        self.agg_configs_.append((cat, num, 'mean'))
                        self.agg_configs_.append((cat, num, 'std'))

            # 4. Target Encoding Setup
            if task in [ProblemType.REGRESSION, ProblemType.BINARY_CLASSIFICATION]:
                self.global_mean_ = y.mean()
                y_series = y.copy()
                for col in cat_cols:
                     if X[col].nunique() > 10:
                         encoded_map = self._fit_target_encode(X[col], y_series)
                         self.target_encodings_[col] = encoded_map

            # --- Apply Transformations ---
            df_trans = self._transform_polars(df)
            
            # Convert back to Pandas
            X_out = df_trans.to_pandas()
            
            # Apply Target Encoding
            for col, map_dict in self.target_encodings_.items():
                if col in X_out.columns:
                     X_out[f'{col}_tenc'] = X_out[col].map(map_dict).fillna(self.global_mean_)

            # 5. Correlation Filtering
            if X_out.shape[1] > self.max_features:
                self.features_to_keep_ = self._filter_correlation(X_out)
                X_out = X_out[self.features_to_keep_]
            else:
                self.features_to_keep_ = X_out.columns.tolist()

            return X_out

        except Exception as e:
            logger.warning(f"AutoFeatureEngineer failed: {e}. Returning original features.")
            self.features_to_keep_ = X.columns.tolist()
            return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = pl.from_pandas(X)
            df_trans = self._transform_polars(df)
            X_out = df_trans.to_pandas()
            
            # Target Encoding
            for col, map_dict in self.target_encodings_.items():
                if col in X_out.columns:
                     X_out[f'{col}_tenc'] = X_out[col].map(map_dict).fillna(self.global_mean_)

            # Ensure all expected columns exist
            for col in self.features_to_keep_:
                if col not in X_out.columns:
                    X_out[col] = 0.0
            
            return X_out[self.features_to_keep_]
        except Exception as e:
            return X

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        
        # Log Transforms
        for col in self.skewed_cols_:
            exprs.append(pl.col(col).fill_null(0).abs().log1p().alias(f"log_{col}"))
            
        # Interactions
        for col1, col2, op in self.interaction_pairs_:
            if op == '*':
                exprs.append((pl.col(col1) * pl.col(col2)).alias(f"{col1}_x_{col2}"))
            elif op == '/':
                # Safe division: A / (B + 1e-6)
                exprs.append((pl.col(col1) / (pl.col(col2) + 1e-6)).alias(f"{col1}_div_{col2}"))
            elif op == '^2':
                exprs.append((pl.col(col1) ** 2).alias(f"{col1}_sq"))
        
        # Grouped Aggregations
        for cat, num, stat in self.agg_configs_:
            if stat == 'mean':
                exprs.append(pl.col(num).mean().over(cat).fill_null(0).alias(f"{cat}_{num}_mean"))
            elif stat == 'std':
                exprs.append(pl.col(num).std().over(cat).fill_null(0).alias(f"{cat}_{num}_std"))
                
        if exprs:
            df = df.with_columns(exprs)
            
        return df

    def _fit_target_encode(self, series: pd.Series, y: pd.Series) -> Dict[str, float]:
        agg = pd.DataFrame({'cat': series, 'y': y}).groupby('cat')['y'].agg(['mean', 'count'])
        smoothing = 10
        agg['smooth'] = (agg['count'] * agg['mean'] + smoothing * self.global_mean_) / (agg['count'] + smoothing)
        return agg['smooth'].to_dict()

    def _filter_correlation(self, X: pd.DataFrame) -> List[str]:
        # Simple correlation filter using Pandas (Polars corr is fast but matrix handling in pandas is easier for drop)
        # Select numeric only
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty: return X.columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = X_num.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        
        return [c for c in X.columns if c not in to_drop]