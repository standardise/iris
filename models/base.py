from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple

class CandidateModel(ABC):
    """
    Abstract Base Class for all models in Iris.
    """
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self._model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, time_limit: int = None):
        """Train the model with optional time budget."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and return numpy array."""
        pass
    
    def explain(self, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Calculate SHAP values. Returns (base_value, {feature: contribution}).
        """
        return 0.0, {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None