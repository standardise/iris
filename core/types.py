from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, ConfigDict

class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"

class FeatureSchema(BaseModel):
    name: str
    dtype: str

class ModelMetrics(BaseModel):
    main_metric: str
    scores: Dict[str, float]

class ModelBlueprint(BaseModel):
    model_version: str = "1.0.0"
    task_type: str
    strategy_used: str
    is_timeseries: bool
    input_features: List[FeatureSchema]
    target_column: str
    active_models: List[str]
    ensemble_weights: Dict[str, float]
    metrics: ModelMetrics
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class PredictionAudit(BaseModel):
    model_version: str
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional['Explanation'] = None

class Explanation(BaseModel):
    base_value: float
    contributions: Dict[str, float]

class LoadMode(str, Enum):
    Default = "default"
    Lazy = "lazy"

class VisualizationType(str, Enum):
    TIME_SERIES = "time_series"
    BAR_CHART = "bar_chart"
    SCATTER = "scatter"
    METRIC_CARD = "metric_card"

class VisualizationData(BaseModel):
    type: VisualizationType
    title: str
    data: List[Dict[str, Any]] # JSON-friendly for frontend (e.g. Recharts)
    axes: Dict[str, str] = {} # e.g. {"x": "date", "y": "sales"}

class InferenceResult(BaseModel):
    prediction: Any # The core result (value, class, cluster ID)
    summary: str # Human-readable explanation
    details: Dict[str, Any] = {} # Advanced metrics/params
    visualization: Optional[VisualizationData] = None # Plotting data
