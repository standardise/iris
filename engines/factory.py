from iris.core.types import ProblemType
from iris.engines.regression import RegressionEngine
from iris.engines.classification import ClassificationEngine
from iris.engines.timeseries import TimeSeriesEngine

class EngineFactory:
    @staticmethod
    def create(task_type: ProblemType, future_steps: int = 1, **kwargs):
        if task_type == ProblemType.REGRESSION:
            return RegressionEngine()
        elif task_type == ProblemType.BINARY_CLASSIFICATION:
            return ClassificationEngine(is_multiclass=False)
        elif task_type == ProblemType.MULTICLASS_CLASSIFICATION:
            return ClassificationEngine(is_multiclass=True)
        elif task_type == ProblemType.TIME_SERIES_FORECASTING:
            return TimeSeriesEngine(
                date_col=kwargs.get('date_col'),
                id_col=kwargs.get('id_col')
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")