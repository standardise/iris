from iris.foundation.types import ProblemType
from iris.engine.regressors import RegressionEngine
from iris.engine.classifiers import ClassificationEngine

class EngineFactory:
    @staticmethod
    def create(task_type: ProblemType, future_steps: int = 1):
        if task_type == ProblemType.REGRESSION:
            return RegressionEngine()
        elif task_type == ProblemType.BINARY_CLASSIFICATION:
            return ClassificationEngine(is_multiclass=False)
        elif task_type == ProblemType.MULTICLASS_CLASSIFICATION:
            return ClassificationEngine(is_multiclass=True)
        else:
            raise ValueError(f"Unknown task type: {task_type}")