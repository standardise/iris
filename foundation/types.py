from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Set
from enum import Enum
import math

class ProblemType(str, Enum):
    """
    Defines the specific nature of the machine learning task.
    Distinguishing Binary vs Multiclass is crucial for selecting the correct loss function.
    """
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"


class Metric(str, Enum):
    """
    Defines supported evaluation metrics.
    Optimization metrics are used by Optuna.
    Reporting metrics are for human readability.
    """
    RMSE = "rmse"            # Root Mean Squared Error (Standard Optimization)
    MAE = "mae"              # Mean Absolute Error (Robust to outliers)
    MAPE = "mape"            # Mean Absolute Percentage Error (Business favorite)
    R2 = "r2"                # R-Squared (Goodness of fit)

    LOG_LOSS = "log_loss"    # Logarithmic Loss (Best for Probabilistic Optimization)
    ROC_AUC = "roc_auc"      # Area Under Curve (Ranking Quality)
    ACCURACY = "accuracy"    # (Reporting Only - not good for optimization)
    F1_SCORE = "f1_score"    # (Reporting Only - essential for imbalanced data)


DEFAULT_METRICS: Dict[ProblemType, Metric] = {
    ProblemType.REGRESSION: Metric.RMSE,
    ProblemType.TIME_SERIES_FORECASTING: Metric.RMSE,
    ProblemType.BINARY_CLASSIFICATION: Metric.LOG_LOSS,
    ProblemType.MULTICLASS_CLASSIFICATION: Metric.LOG_LOSS
}

SUPPORTED_METRICS: Dict[ProblemType, Set[Metric]] = {
    ProblemType.REGRESSION: {
        Metric.RMSE, 
        Metric.MAE, 
        Metric.MAPE, 
        Metric.R2
    },
    ProblemType.TIME_SERIES_FORECASTING: {
        Metric.RMSE, 
        Metric.MAE, 
        Metric.MAPE, 
        Metric.R2
    },
    ProblemType.BINARY_CLASSIFICATION: {
        Metric.LOG_LOSS, 
        Metric.ROC_AUC, 
        Metric.ACCURACY, 
        Metric.F1_SCORE
    },
    ProblemType.MULTICLASS_CLASSIFICATION: {
        Metric.LOG_LOSS, 
        Metric.ACCURACY, 
        Metric.F1_SCORE
    }
}

class ProbelmValidator:
    """Helper methods for Type safety checks."""
    
    @staticmethod
    def validate_metric(task: ProblemType, metric: Metric) -> None:
        """
        Raises ValueError if the metric is not supported for the given task.
        """
        if metric not in SUPPORTED_METRICS[task]:
            raise ValueError(
                f"Metric '{metric.value}' is not supported for task '{task.value}'. "
                f"Supported metrics: {[m.value for m in SUPPORTED_METRICS[task]]}"
            )

    @staticmethod
    def is_classification(task: ProblemType) -> bool:
        """Check if task uses Classifiers (Categorical Output)"""
        return task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}

    @staticmethod
    def is_regression(task: ProblemType) -> bool:
        """
        Check if task uses Regressors (Numerical Output).
        Note: Time Series is mathematically a regression task.
        """
        return task in {ProblemType.REGRESSION, ProblemType.TIME_SERIES_FORECASTING}
    
    @staticmethod
    def is_timeseries(task: ProblemType) -> bool:
        """
        Check if task requires temporal handling (Sorting, No Shuffle, Lags).
        Use this to trigger TimeSeriesSplit and Lag Feature generation.
        """
        return task == ProblemType.TIME_SERIES_FORECASTING

class LoadMode(str, Enum):
    Default = "default"
    Lazy = "lazy"

class FeatureSchema(BaseModel):
    name: str
    dtype: str 

class ModelMetrics(BaseModel):
    main_metric: str
    scores: Dict[str, float]

class Explanation(BaseModel):
    """
    Structured Explainability Data (SHAP-based).
    ต้องมี base_value เพื่อให้วาดกราฟ Waterfall ได้ถูกต้อง
    """
    base_value: float = Field(..., description="The intercept or average value")
    contributions: Dict[str, float] = Field(..., description="Feature contribution scores")

class ModelBlueprint(BaseModel):
    model_version: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    created_at: datetime = Field(default_factory=datetime.now)
    
    task_type: str
    strategy_used: str
    is_timeseries: bool
    
    input_features: List[FeatureSchema]
    target_column: str
    
    active_models: List[str]
    ensemble_weights: Dict[str, float]
    metrics: ModelMetrics

    @field_validator('ensemble_weights')
    def enforce_math_integrity(cls, v):
        """
        ตรวจสอบและ Normalize weight ให้รวมกันได้ 1.0 เสมอ
        """
        if any(w < 0 for w in v.values()):
            raise ValueError("Ensemble weights cannot be negative.")

        total = sum(v.values())
        if total == 0:
            raise ValueError("Total weight cannot be zero.")

        if not math.isclose(total, 1.0, rel_tol=1e-3):
            return {k: val / total for k, val in v.items()}
        
        return v

    class Config:
        frozen = True

class PredictionAudit(BaseModel):
    model_version: str
    prediction: Union[float, int, str] 
    probabilities: Optional[Dict[str, float]] = None 
    explanation: Optional[Explanation] = None