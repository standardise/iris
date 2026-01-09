import logging
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

from iris.dataset import Dataset
from iris.foundation.types import ModelBlueprint, FeatureSchema, ModelMetrics
from iris.foundation.types import ProblemType
from iris.engine.interfaces import CandidateModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from iris.feature_engineering.automated import AutoFeatureEngineer
except ImportError:
    AutoFeatureEngineer = None

class BaseEngine(ABC):
    """
    The Orchestrator for the AutoML process.

    Responsibilities:
    - Automated Feature Engineering execution.
    - Dynamic Time Budget Management for each model.
    - Safe Training Loop (catching errors per model).
    - Ensemble Weight Optimization using SLSQP.

    Attributes:
        task (ProblemType): The type of machine learning problem (Regression/Classification).
        candidates (List[CandidateModel]): List of models to be trained.
        trained_models (Dict[str, CandidateModel]): Dictionary of successfully trained models.
        model_weights (Dict[str, float]): Ensemble weights for each model.
    """

    def __init__(self, task: ProblemType):
        self.task = task
        self.candidates: List[CandidateModel] = []
        self.trained_models: Dict[str, CandidateModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.feature_dtypes = {}
        self.feature_engineer: Optional[Any] = None 

    def register_candidates(self, models: List[CandidateModel]):
        """Registers a list of candidate models to be trained."""
        self.candidates.extend(models)

    def fit(self, dataset: Dataset, time_limit: int = 300) -> ModelBlueprint:
        """
        Orchestrates the training pipeline: Feature Engineering -> Split -> Train -> Optimize -> Refit.

        Args:
            dataset (Dataset): The training data.
            time_limit (int): Global time budget in seconds.

        Returns:
            ModelBlueprint: The training summary and artifacts.

        Raises:
            RuntimeError: If no models could be trained successfully.
        """
        start_time = time.time()
        X, y = dataset.get_X_y()
        self.feature_dtypes = dataset.features.dtypes 
        
        is_tabular = self.task in [ProblemType.REGRESSION, ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]
        
        if is_tabular and time_limit > 60 and AutoFeatureEngineer is not None:
            try:
                logger.info("Starting Auto Feature Engineering...")
                self.feature_engineer = AutoFeatureEngineer()
                X = self.feature_engineer.fit_transform(X, y, self.task)
                logger.info(f"Feature Engineering completed. New feature count: {X.shape[1]}")
            except Exception as e:
                logger.warning(f"Feature Engineering failed. Proceeding with raw features. Reason: {e}")
                self.feature_engineer = None

        stratify = y if self.task in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION] else None
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        logger.info(f"Engine execution started. Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples.")

        if not self.candidates: 
            raise RuntimeError("No models registered. Please register models via the EngineFactory.")
        
        val_predictions = {}
        remaining_models = len(self.candidates)
        
        for i, model in enumerate(self.candidates):
            elapsed = time.time() - start_time
            remaining_total_time = time_limit - elapsed
            
            if remaining_total_time < 5:
                logger.warning(f"Global time limit reached. Skipping remaining {remaining_models - i} models.")
                break
            
            models_left = remaining_models - i
            current_budget = remaining_total_time / models_left
            
            try:
                model.fit(X_train, y_train, time_limit=int(current_budget)) 
                
                pred_val = model.predict(X_val)
                val_predictions[model.name] = pred_val
                self.trained_models[model.name] = model
                
            except Exception as e:
                logger.error(f"Model '{model.name}' failed to train. Reason: {e}")

        if not self.trained_models:
            raise RuntimeError("All candidate models failed to train. Please check data quality or constraints.")

        logger.info("Optimizing ensemble weights using SLSQP...")
        self.model_weights, best_score = self._optimize_weights(y_val, val_predictions)

        logger.info("Refitting active models on the full dataset...")
        for name, weight in self.model_weights.items():
            if weight > 0:
                model = self.trained_models[name]
                try:
                    model.fit(X, y, time_limit=time_limit) 
                except Exception as e:
                    logger.warning(f"Refit failed for {name}: {e}. Keeping the validation-trained version.")
        
        return self._create_blueprint(dataset, best_score)

    def predict(self, X: pd.DataFrame) -> Any:
        """
        Predicts target values for new data using the weighted ensemble.
        
        Args:
            X (pd.DataFrame): Input features.

        Returns:
            Any: Combined predictions (np.ndarray or pd.Series).
        """
        X_processed = X.copy()
        
        if self.feature_engineer:
            try:
                X_processed = self.feature_engineer.transform(X)
            except Exception as e:
                logger.warning(f"Feature transformation failed during prediction: {e}. Falling back to raw features.")
                pass 

        preds = {}
        for name, weight in self.model_weights.items():
            if weight > 0 and name in self.trained_models:
                preds[name] = self.trained_models[name].predict(X_processed)
        
        return self._ensemble_predict(preds)

    def explain(self, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Calculates Feature Importance using Weighted Average SHAP values.
        
        Args:
            X (pd.DataFrame): Input features to explain.

        Returns:
            Tuple[float, Dict[str, float]]: Base value and Dictionary of {feature: contribution}.
        """
        X_processed = X.copy()
        if self.feature_engineer:
            try:
                X_processed = self.feature_engineer.transform(X)
            except Exception:
                pass

        total_base = 0.0
        total_contribs = {col: 0.0 for col in X_processed.columns}
        
        for name, weight in self.model_weights.items():
            if weight > 0 and name in self.trained_models:
                try:
                    base, contribs = self.trained_models[name].explain(X_processed)
                    total_base += (base * weight)
                    for feature, val in contribs.items():
                        if feature in total_contribs:
                            total_contribs[feature] += (val * weight)
                except Exception as e:
                    logger.debug(f"Could not explain model {name}: {e}")
        
        return total_base, total_contribs

    def _ensemble_predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        if not preds: raise RuntimeError("No predictions available for ensemble.")
        
        first_pred = next(iter(preds.values()))
        final_pred = np.zeros_like(first_pred)
        
        for name, pred in preds.items():
            weight = self.model_weights.get(name, 0.0)
            final_pred += (pred * weight)
        return final_pred

    def _optimize_weights(self, y_true: pd.Series, preds: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], float]:
        """
        Internal: Finds optimal ensemble weights using SLSQP to minimize loss.
        """
        model_names = list(preds.keys())
        
        try:
            pred_matrix = np.column_stack([preds[name] for name in model_names])
        except ValueError:
             return self._fallback_optimize_weights(y_true, preds)

        y_true_np = y_true.values
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}

        def loss_func(weights):
            final_pred = np.dot(pred_matrix, weights)
            if is_classification:
                final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)
                return log_loss(y_true_np, final_pred)
            else:
                return mean_squared_error(y_true_np, final_pred)

        constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
        bounds = [(0.0, 1.0)] * len(model_names)
        init_guess = [1.0 / len(model_names)] * len(model_names)

        try:
            res = minimize(loss_func, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-4)
            best_weights = res.x
            best_score = res.fun
        except Exception as e:
            logger.warning(f"SLSQP Optimization failed: {e}. Using fallback weights.")
            return self._fallback_optimize_weights(y_true, preds)

        weights_dict = {name: float(w) for name, w in zip(model_names, best_weights)}
        weights_dict = {k: v for k, v in weights_dict.items() if v > 0.01}
        
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v/total for k, v in weights_dict.items()}
        else:
            return self._fallback_optimize_weights(y_true, preds)

        if not is_classification:
            best_score = np.sqrt(best_score)

        return weights_dict, best_score

    def _fallback_optimize_weights(self, y_true, preds) -> Tuple[Dict[str, float], float]:
        """Simple inverse-error weighting if optimization fails."""
        scores = {}
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        
        for name, pred in preds.items():
            try:
                if is_classification: err = log_loss(y_true, pred)
                else: err = mean_squared_error(y_true, pred)
            except: err = float('inf')
            
            scores[name] = 1 / (err + 1e-9)
            
        total = sum(scores.values())
        if total == 0: 
            return {k: 1.0/len(scores) for k in scores}, 0.0
        
        weights = {k: v/total for k, v in scores.items()}
        return weights, 0.0

    def _create_blueprint(self, dataset: Dataset, score: float) -> ModelBlueprint:
        if self.task == ProblemType.REGRESSION: metric_name = "rmse"
        elif self.task == ProblemType.TIME_SERIES_FORECASTING: metric_name = "rmse"
        else: metric_name = "log_loss"

        return ModelBlueprint(
            task_type=self.task.value,
            strategy_used=self.__class__.__name__,
            is_timeseries=(dataset.date_col is not None),
            input_features=[FeatureSchema(name=c, dtype=str(t)) for c, t in self.feature_dtypes.items()],
            target_column=dataset.target_name,
            active_models=list(self.model_weights.keys()),
            ensemble_weights=self.model_weights,
            metrics=ModelMetrics(main_metric=metric_name, scores={"val_score": float(score)})
        )