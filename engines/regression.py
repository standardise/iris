from iris.engines.base import BaseEngine
from iris.core.types import ProblemType
from iris.models.supervised import RidgeRegressorModel, LGBMRegressorModel, CatBoostRegressorModel, HistGBRegressorModel

class RegressionEngine(BaseEngine):
    def __init__(self):
        super().__init__(task=ProblemType.REGRESSION)

    def fit(self, dataset, time_limit=300):
        self.candidates = []
        self.register_candidates([
            LGBMRegressorModel("LGBM_Accurate", mode="accurate"),
            CatBoostRegressorModel("CatBoost_Balanced", mode="balanced"),
            RidgeRegressorModel("Ridge")
        ])
            
        return super().fit(dataset, time_limit)