import pandas as pd
import numpy as np
import logging
from typing import Literal, Dict, List, Optional, Union

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from iris.dataset import Dataset
from iris.features.automated import AutoFeatureEngineer
from iris.core.types import InferenceResult, VisualizationData, VisualizationType

logger = logging.getLogger(__name__)

class UnsupervisedEngine:
    """
    Engine for Unsupervised Learning tasks: Clustering, Similarity, Anomaly Detection.
    """
    
    def __init__(self, task: Literal['clustering', 'similarity', 'anomaly']):
        self.task = task
        self.model = None
        self.feature_engineer = None
        self.preprocessor = None
        self.feature_names_ = []
        self._X_train = None 

    def fit(self, dataset: Dataset, n_clusters: int = 5):
        logger.info(f"Starting Unsupervised Training: {self.task}")
        X = dataset.features.copy()
        
        X_num = X.select_dtypes(include=[np.number])
        X_num = X_num.fillna(0)
        
        self.feature_names_ = X_num.columns.tolist()
        
        self.preprocessor = make_pipeline(StandardScaler())
        X_vec = self.preprocessor.fit_transform(X_num)
        
        if self.task == 'clustering':
            logger.info(f"Clustering with K-Means (k={n_clusters})...")
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            self.model.fit(X_vec)
            
        elif self.task == 'similarity':
            logger.info("Indexing data for Similarity Search (KNN)...")
            self.model = NearestNeighbors(n_neighbors=10, algorithm='auto')
            self.model.fit(X_vec)
            self._X_train = X_vec 
            
        elif self.task == 'anomaly':
            from sklearn.ensemble import IsolationForest
            logger.info("Detecting Anomalies with Isolation Forest...")
            self.model = IsolationForest(random_state=42, n_jobs=-1)
            self.model.fit(X_vec)

        return self

    def predict(self, dataset: Dataset) -> Union[pd.Series, pd.DataFrame]:
        X = dataset.features.copy()
        X_num = X[self.feature_names_].fillna(0)
        X_vec = self.preprocessor.transform(X_num)
        
        if self.task == 'clustering':
            labels = self.model.predict(X_vec)
            return pd.Series(labels, name="cluster")
            
        elif self.task == 'anomaly':
            scores = self.model.decision_function(X_vec)
            labels = self.model.predict(X_vec)
            return pd.DataFrame({'anomaly_score': scores, 'is_anomaly': labels == -1})
            
        elif self.task == 'similarity':
            raise ValueError("For similarity, use 'query()' method.")

    def predict_response(self, dataset: Dataset) -> InferenceResult:
        """
        Generates a rich prediction response.
        """
        X = dataset.features.copy()
        X_num = X[self.feature_names_].fillna(0)
        X_vec = self.preprocessor.transform(X_num)
        
        if self.task == 'clustering':
            labels = self.model.predict(X_vec)
            lbl = labels[0] # Assume single row for rich response context
            
            return InferenceResult(
                prediction=int(lbl),
                summary=f"Assigned to Cluster {lbl}",
                details={},
                visualization=None
            )
            
        elif self.task == 'anomaly':
            scores = self.model.decision_function(X_vec)
            labels = self.model.predict(X_vec)
            
            score = scores[0]
            is_anom = labels[0] == -1
            status = "ANOMALY" if is_anom else "Normal"
            
            viz = VisualizationData(
                type=VisualizationType.METRIC_CARD,
                title="Anomaly Check",
                data=[
                    {"label": "Status", "value": status},
                    {"label": "Score", "value": float(score)}
                ]
            )
            
            return InferenceResult(
                prediction="Anomaly" if is_anom else "Normal",
                summary=f"Detected: {status} (Score: {score:.3f})",
                details={"score": float(score)},
                visualization=viz
            )
            
        else:
            raise ValueError("For similarity, use 'query_similarity()' instead.")

    def query(self, dataset: Dataset, k: int = 5) -> Dict[int, List[int]]:
        if self.task != 'similarity':
            raise ValueError("Engine is not in similarity mode.")
            
        X = dataset.features.copy()
        X_num = X[self.feature_names_].fillna(0)
        X_vec = self.preprocessor.transform(X_num)
        
        distances, indices = self.model.kneighbors(X_vec, n_neighbors=k)
        
        results = {}
        for i, neighbor_indices in enumerate(indices):
            results[i] = neighbor_indices.tolist()
            
        return results