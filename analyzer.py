from typing import Optional, Union, List, Dict
import pandas as pd
import logging
from iris.dataset import Dataset
from iris.engines.unsupervised import UnsupervisedEngine

class Analyzer:
    """
    Public API for Unsupervised Learning and Data Analysis.
    Supports: Clustering, Similarity Search, Anomaly Detection.
    """
    
    def __init__(self, task: str = 'clustering', verbose: bool = True):
        if task not in ['clustering', 'similarity', 'anomaly']:
            raise ValueError("Task must be one of: 'clustering', 'similarity', 'anomaly'")
            
        self.task = task
        self.verbose = verbose
        self._engine = UnsupervisedEngine(task)
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

    def fit(self, dataset: Dataset, n_clusters: int = 5):
        """Trains the analyzer on the dataset."""
        self._engine.fit(dataset, n_clusters=n_clusters)
        if self.verbose:
            logging.info("Analysis model trained successfully.")

    def get_clusters(self, dataset: Dataset) -> pd.DataFrame:
        """Returns the dataset with an added 'cluster' column."""
        if self.task != 'clustering':
            raise RuntimeError("Analyzer is not in clustering mode.")
            
        clusters = self._engine.predict(dataset)
        res = dataset.df.copy()
        res['cluster'] = clusters.values
        return res

    def get_anomalies(self, dataset: Dataset) -> pd.DataFrame:
        """Returns the dataset with 'anomaly_score' and 'is_anomaly' columns."""
        if self.task != 'anomaly':
            raise RuntimeError("Analyzer is not in anomaly mode.")
            
        anom_info = self._engine.predict(dataset)
        res = dataset.df.copy()
        res = pd.concat([res.reset_index(drop=True), anom_info.reset_index(drop=True)], axis=1)
        return res

    def query_similarity(self, query_data: Union[Dataset, pd.DataFrame], k: int = 5) -> Dict[int, List[int]]:
        """
        Finds similar items from the training set.
        
        Args:
            query_data: The item(s) to find matches for.
            k: Number of matches to return.
            
        Returns:
            Dict where key is the index of the query item, and value is a list of 
            indices from the training set that are most similar.
        """
        if self.task != 'similarity':
            raise RuntimeError("Analyzer is not in similarity mode.")
            
        if isinstance(query_data, pd.DataFrame):
            # Hack: Create a temporary dataset. Target is dummy.
            ds = Dataset(src=query_data, target=query_data.columns[0]) 
        else:
            ds = query_data
            
        return self._engine.query(ds, k=k)
