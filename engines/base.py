import logging
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.linear_model import Ridge, LogisticRegression

from iris.dataset import Dataset
from iris.core.types import ModelBlueprint, FeatureSchema, ModelMetrics, InferenceResult, ContextData, ContextType
from iris.core.types import ProblemType
from iris.models.base import CandidateModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from iris.features.automated import AutoFeatureEngineer
except ImportError:
    AutoFeatureEngineer = None

class BaseEngine(ABC):
    """
    The Orchestrator for the AutoML process.
    """

    def __init__(self, task: ProblemType):
        self.task = task
        self.candidates: List[CandidateModel] = []
        self.trained_models: Dict[str, CandidateModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.feature_dtypes = {}
        self.feature_engineer: Optional[Any] = None 
        self.meta_model: Optional[Any] = None
        self.stacking_active: bool = False
        self.train_stats: Dict[str, Any] = {} # For contextual predictions

    def register_candidates(self, models: List[CandidateModel]):
        self.candidates.extend(models)

    def _fit_candidates_and_ensemble(self, X_train, y_train, X_val, y_val, time_limit, start_time):
        """
        Internal method to train candidates, optimize weights, and try stacking.
        Enforces column sorting for consistency.
        """
        X_train = X_train.reindex(sorted(X_train.columns), axis=1)
        X_val = X_val.reindex(sorted(X_val.columns), axis=1)
        
        val_predictions = {}
        
        for i, model in enumerate(self.candidates):
            elapsed = time.time() - start_time
            remaining_total_time = time_limit - elapsed
            
            if remaining_total_time < 5:
                break
            
            models_left = len(self.candidates) - i
            current_budget = remaining_total_time / models_left
            
            try:
                model.fit(X_train, y_train, time_limit=int(current_budget)) 
                
                pred_val = model.predict(X_val)
                val_predictions[model.name] = pred_val
                self.trained_models[model.name] = model
                
            except Exception as e:
                logger.error(f"Model '{model.name}' failed to train. Reason: {e}")

        if not self.trained_models:
            raise RuntimeError("All candidate models failed to train.")

        logger.info("Optimizing ensemble weights using SLSQP...")
        self.model_weights, weighted_score = self._optimize_weights(y_val, val_predictions)
        
        # --- Stacking (Blended Ensemble) ---
        model_names = list(val_predictions.keys())
        first_pred = val_predictions[model_names[0]]
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        
        X_meta = None
        if is_classification and first_pred.ndim == 2:
             X_meta = np.hstack([val_predictions[name] for name in model_names])
        else:
             X_meta = np.column_stack([val_predictions[name] for name in model_names])
             
        meta_score = float('inf')
        best_score = weighted_score
        
        try:
            if is_classification:
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(X_meta, y_val)
                meta_pred = self.meta_model.predict_proba(X_meta)
                if first_pred.ndim == 1: meta_pred = meta_pred[:, 1]
                meta_pred = np.clip(meta_pred, 1e-15, 1 - 1e-15)
                meta_score = log_loss(y_val, meta_pred)
            else:
                self.meta_model = Ridge(alpha=1.0)
                self.meta_model.fit(X_meta, y_val)
                meta_pred = self.meta_model.predict(X_meta)
                meta_score = np.sqrt(mean_squared_error(y_val, meta_pred))
                
            logger.info(f"Stacking Score: {meta_score:.4f} vs Weighted Score: {weighted_score:.4f}")
            
            if meta_score < weighted_score:
                self.stacking_active = True
                best_score = meta_score
                logger.info(">> Stacking Strategy WON. Using Meta-Learner.")
            else:
                self.stacking_active = False
                self.meta_model = None
                logger.info(">> Weighted Ensemble WON. Using SLSQP Weights.")
                
        except Exception as e:
            logger.warning(f"Stacking training failed: {e}. Fallback to Weighted Ensemble.")
            self.stacking_active = False
            
        return best_score

    def fit(self, dataset: Dataset, time_limit: int = 300) -> ModelBlueprint:
        start_time = time.time()
        X, y = dataset.get_X_y()
        self.feature_dtypes = dataset.features.dtypes 
        
        # Store Stats for Rich Output
        if self.task == ProblemType.REGRESSION:
            self.train_stats = {
                "mean": float(y.mean()),
                "min": float(y.min()),
                "max": float(y.max()),
                "std": float(y.std())
            }
        elif self.task in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            self.train_stats = {
                "classes": sorted(y.unique().tolist()),
                "counts": y.value_counts().to_dict()
            }
        
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

        # --- Large Dataset Optimization ---
        if len(X_train) > 50000:
            pass

        if not self.candidates: 
            raise RuntimeError("No models registered. Please register models via the EngineFactory.")
        
        best_score = self._fit_candidates_and_ensemble(X_train, y_train, X_val, y_val, time_limit, start_time)

        elapsed_so_far = time.time() - start_time
        remaining_time = time_limit - elapsed_so_far
        
        if self.stacking_active:
            active_models = list(self.trained_models.keys())
        else:
            active_models = [name for name, w in self.model_weights.items() if w > 0]
        
        if active_models and remaining_time > 5:
            logger.info(f"Refitting {len(active_models)} active models on full dataset (Time left: {remaining_time:.1f}s)...")
            X = X.reindex(sorted(X.columns), axis=1)
            time_per_model = remaining_time / len(active_models)
            
            for name in active_models:
                model = self.trained_models[name]
                try:
                    budget = max(5, int(time_per_model))
                    model.fit(X, y, time_limit=budget) 
                except Exception as e:
                    logger.warning(f"Refit failed for {name}: {e}. Keeping the validation-trained version.")
        else:
            logger.info("Skipping refit due to time constraints. Using validation-trained models.")
        
        return self._create_blueprint(dataset, best_score)

    def predict(self, X: pd.DataFrame) -> Any:
        X_processed = X.copy()
        
        if self.feature_engineer:
            try:
                X_processed = self.feature_engineer.transform(X)
            except Exception as e:
                pass 

        X_processed = X_processed.reindex(sorted(X_processed.columns), axis=1)

        preds = {}
        target_models = self.trained_models.keys() if self.stacking_active else [n for n, w in self.model_weights.items() if w > 0]
        
        for name in target_models:
            if name in self.trained_models:
                preds[name] = self.trained_models[name].predict(X_processed)
        
        if self.stacking_active:
            return self._stacking_predict(preds)
        else:
            return self._ensemble_predict(preds)

    def predict_response(self, X: pd.DataFrame) -> InferenceResult:
        """Generates a rich prediction response with data context."""
        is_classification = self.task in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]
        
        if is_classification:
            probs = self.predict(X) 
            n_rows = len(X)
            if n_rows > 1:
                return InferenceResult(
                    prediction=probs.tolist() if isinstance(probs, np.ndarray) else probs,
                    summary=f"Processed {n_rows} samples.",
                    details={}
                )
            
            row_prob = probs[0] if probs.ndim == 2 else probs
            classes = self.train_stats.get('classes', [i for i in range(len(row_prob))])
            
            top_idx = np.argmax(row_prob)
            top_class = classes[top_idx]
            top_conf = row_prob[top_idx]
            
            summary = f"Predicted: {top_class} ({top_conf:.1%})"
            
            chart_data = []
            for cls, p in zip(classes, row_prob):
                chart_data.append({"label": str(cls), "value": float(p)})
                
            ctx = ContextData(
                type=ContextType.DISTRIBUTION,
                title="Class Probabilities",
                data=chart_data,
                axes={"x": "Class", "y": "Probability"}
            )
            
            return InferenceResult(
                prediction=str(top_class),
                summary=summary,
                details={"confidence": float(top_conf)},
                context=ctx
            )
            
        else: # Regression
            preds = self.predict(X)
            val = preds[0]
            mean_val = self.train_stats.get('mean', 0)
            
            diff = val - mean_val
            direction = "above" if diff > 0 else "below"
            pct = (abs(diff) / (mean_val + 1e-9)) * 100
            
            summary = f"Predicted: {val:,.2f} ({pct:.1f}% {direction} average)"
            
            ctx = ContextData(
                type=ContextType.METRIC,
                title="Prediction vs Average",
                data=[
                    {"label": "Prediction", "value": float(val)},
                    {"label": "Global Average", "value": float(mean_val)}
                ]
            )
            
            return InferenceResult(
                prediction=float(val),
                summary=summary,
                details={"deviation_from_mean": float(diff)},
                context=ctx
            )

    def explain(self, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        X_processed = X.copy()
        if self.feature_engineer:
            try:
                X_processed = self.feature_engineer.transform(X)
            except Exception: pass

        X_processed = X_processed.reindex(sorted(X_processed.columns), axis=1)

        total_base = 0.0
        total_contribs = {col: 0.0 for col in X_processed.columns}
        weights = self.model_weights 
        
        for name, weight in weights.items():
            if weight > 0 and name in self.trained_models:
                try:
                    base, contribs = self.trained_models[name].explain(X_processed)
                    total_base += (base * weight)
                    for feature, val in contribs.items():
                        if feature in total_contribs:
                            total_contribs[feature] += (val * weight)
                except Exception as e:
                    pass
        return total_base, total_contribs

    def _ensemble_predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        if not preds: raise RuntimeError("No predictions available.")
        first_pred = next(iter(preds.values()))
        final_pred = np.zeros_like(first_pred)
        for name, pred in preds.items():
            weight = self.model_weights.get(name, 0.0)
            final_pred += (pred * weight)
        return final_pred

    def _stacking_predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        model_names = list(preds.keys()) 
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        first_pred = next(iter(preds.values()))
        X_meta = None
        if is_classification and first_pred.ndim == 2:
             X_meta = np.hstack([preds[name] for name in model_names])
        else:
             X_meta = np.column_stack([preds[name] for name in model_names])
        if is_classification:
            final_prob = self.meta_model.predict_proba(X_meta)
            if first_pred.ndim == 1: return final_prob[:, 1]
            return final_prob
        else:
            return self.meta_model.predict(X_meta)

    def _optimize_weights(self, y_true: pd.Series, preds: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], float]:
        model_names = list(preds.keys())
        y_true_np = y_true.values
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        try:
            first_pred = preds[model_names[0]]
            if is_classification and first_pred.ndim == 2:
                pred_tensor = np.stack([preds[name] for name in model_names], axis=-1)
            else:
                pred_tensor = np.column_stack([preds[name] for name in model_names])
        except ValueError: return self._fallback_optimize_weights(y_true, preds)

        def loss_func(weights):
            final_pred = np.dot(pred_tensor, weights)
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
            return {name: float(w)/sum(res.x) for name, w in zip(model_names, res.x)}, res.fun
        except Exception:
            return self._fallback_optimize_weights(y_true, preds)

    def _fallback_optimize_weights(self, y_true, preds) -> Tuple[Dict[str, float], float]:
        scores = {}
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        calc_score = 0.0
        for name, pred in preds.items():
            try:
                if is_classification: 
                    safe_pred = np.clip(pred, 1e-15, 1 - 1e-15)
                    err = log_loss(y_true, safe_pred)
                else: 
                    err = mean_squared_error(y_true, pred)
            except: err = float('inf')
            scores[name] = 1 / (err + 1e-9)
            if calc_score == 0.0: calc_score = err
        total = sum(scores.values())
        if total == 0: return {k: 1.0/len(scores) for k in scores}, calc_score
        return {k: v/total for k, v in scores.items()}, calc_score

    def _create_blueprint(self, dataset: Dataset, score: float) -> ModelBlueprint:
        if self.task == ProblemType.REGRESSION: metric_name = "rmse"
        elif self.task == ProblemType.TIME_SERIES_FORECASTING: metric_name = "rmse"
        else: metric_name = "log_loss"
        return ModelBlueprint(
            task_type=self.task.value,
            strategy_used="Stacking" if self.stacking_active else "WeightedEnsemble",
            is_timeseries=(dataset.date_col is not None),
            input_features=[FeatureSchema(name=c, dtype=str(t)) for c, t in self.feature_dtypes.items()],
            target_column=dataset.target_name,
            active_models=list(self.trained_models.keys() if self.stacking_active else self.model_weights.keys()),
            ensemble_weights=self.model_weights,
            metrics=ModelMetrics(main_metric=metric_name, scores={"val_score": float(score)})
        )