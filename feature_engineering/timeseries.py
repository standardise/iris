import pandas as pd
import numpy as np
import polars as pl
from typing import List, Optional

class TimeSeriesFeatureEngineer:
    """
    Specialized Feature Engineer for Time Series.
    - Extracts Date Parts (Day, Month, Week, etc.)
    - Handles Categorical Columns (ID, Store, Region)
    - Generates Fourier Terms for Seasonality
    """
    
    def __init__(self, date_col: str, id_col: Optional[str] = None):
        self.date_col = date_col
        self.id_col = id_col
        self.feature_names_: List[str] = []
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert to Polars for speed
        try:
            pl_df = pl.from_pandas(df)
        except:
            # Fallback if already polars or weird format
            pl_df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(pd.DataFrame(df))

        # Ensure date column is datetime
        pl_df = pl_df.with_columns(pl.col(self.date_col).cast(pl.Datetime))
        
        # 1. Date Parts Extraction
        # These are safe for gaps and future prediction
        pl_df = pl_df.with_columns([
            pl.col(self.date_col).dt.year().alias("year"),
            pl.col(self.date_col).dt.month().alias("month"),
            pl.col(self.date_col).dt.day().alias("day"),
            pl.col(self.date_col).dt.weekday().alias("day_of_week"),
            pl.col(self.date_col).dt.ordinal_day().alias("day_of_year"),
            pl.col(self.date_col).dt.quarter().alias("quarter"),
            # Cyclical Encoding (Crucial for seasonality)
            (np.pi * 2 * pl.col(self.date_col).dt.month() / 12).sin().alias("month_sin"),
            (np.pi * 2 * pl.col(self.date_col).dt.month() / 12).cos().alias("month_cos"),
        ])
        
        cat_cols = [c for c, t in zip(pl_df.columns, pl_df.dtypes) 
                   if t in [pl.Utf8, pl.Object, pl.Categorical] and c != self.date_col]
        
        min_date = pl_df[self.date_col].min()
        pl_df = pl_df.with_columns(
            ((pl.col(self.date_col) - min_date).dt.total_days()).alias("time_idx")
        )

        res_df = pl_df.to_pandas()
        
        self.feature_names_ = [c for c in res_df.columns if c != self.date_col]
        return res_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Re-apply same logic
        return self.fit_transform(df)