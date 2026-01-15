import pandas as pd
from typing import Union, Optional, Tuple
from iris.core.types import ProblemType
from iris.core.utils import determine_task_type, load_dataset
from iris.core.exception import DataError

class Dataset:
    def __init__(
        self, 
        src: Union[str, pd.DataFrame], 
        target: str, 
        date_column: Optional[str] = None, 
        id_column: Optional[str] = None
    ):
        
        self.src = src
        self.target_name = target
        self.date_col = date_column
        self.id_col = id_column
        self.df = self._load(src)
        self._validate_columns()
        self._preprocess_structural()

        is_ts = (self.date_col is not None)
        self.task_type: ProblemType = determine_task_type(self.df[self.target_name], is_timeseries=is_ts)

    @property
    def features(self) -> pd.DataFrame:
        cols_to_drop = [self.target_name]
        
        # Only drop ID column if it's NOT a Time Series task
        # In Time Series, ID column is used to group series (e.g. Store_ID)
        if self.task_type != ProblemType.TIME_SERIES_FORECASTING:
            if self.id_col and self.id_col in self.df.columns:
                cols_to_drop.append(self.id_col)
                
        return self.df.drop(columns=cols_to_drop)

    @property
    def target(self) -> pd.Series:
        return self.df[self.target_name]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.df.shape

    def get_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.features, self.target

    def _load(self, src) -> pd.DataFrame:
        return load_dataset(src, auto_clean=True)

    def _validate_columns(self):
        if self.df.empty:
            raise DataError("Dataset is empty.")
            
        required_cols = {self.target_name}
        if self.date_col: required_cols.add(self.date_col)
        if self.id_col: required_cols.add(self.id_col)
        
        missing = required_cols - set(self.df.columns)
        if missing:
            raise DataError(f"Missing required columns: {missing}")

    def _preprocess_structural(self):
        """Structural Preprocessing specific to Iris logic (Time/ID)"""
        
        if self.date_col:
            try:
                self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
                self.df = self.df.sort_values(by=self.date_col)
                self.df.set_index(self.date_col, inplace=True, drop=False)
            except Exception as e:
                raise DataError(f"Could not convert '{self.date_col}' to datetime: {e}")

        if self.id_col:
            self.df[self.id_col] = self.df[self.id_col].astype(str)
    
    def __str__(self):
        return f"Dataset(shape={self.shape}, target='{self.target_name}', task_type={self.task_type})"
    
    def __repr__(self):
        return self.__str__()