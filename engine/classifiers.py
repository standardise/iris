from iris.engine.base import BaseEngine
from iris.foundation.types import ProblemType
from iris.engine.models import LogisticClassifierModel, LGBMClassifierModel, CatBoostClassifierModel, HistGBClassifierModel 

class ClassificationEngine(BaseEngine):
    def __init__(self, is_multiclass=False):
        task = ProblemType.MULTICLASS_CLASSIFICATION if is_multiclass else ProblemType.BINARY_CLASSIFICATION
        super().__init__(task=task)

    def fit(self, dataset, time_limit=300):
        self.candidates = []
        
        self.register_candidates([
            LGBMClassifierModel("LGBM_Accurate", mode="accurate"),
            CatBoostClassifierModel("CatBoost_Balanced", mode="balanced"),
            LogisticClassifierModel("Logistic")
        ])
            
        return super().fit(dataset, time_limit)