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
from iris.core.types import ModelBlueprint, FeatureSchema, ModelMetrics
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

    Responsibilities:
    - Automated Feature Engineering execution.
    - Dynamic Time Budget Management for each model.
    - Safe Training Loop (catching errors per model).
    - Ensemble Weight Optimization using SLSQP.
    - Stacking (Blended Ensemble).

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
        self.meta_model: Optional[Any] = None
        self.stacking_active: bool = False

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

        # --- Large Dataset Optimization: Candidate Selection ---
        if len(X_train) > 50000:
            logger.info("Large dataset detected (>50k samples). Running Candidate Selection on subsample...")
            try:
                # Subsample 50k
                sub_size = 50000
                if self.task in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
                    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=sub_size, stratify=y_train, random_state=42)
                else:
                    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=sub_size, random_state=42)
                
                selection_scores = {}
                # Allocate 30% of budget or at least 20s for selection
                sel_budget = max(int(time_limit * 0.3), 20)
                budget_per_model = max(int(sel_budget / len(self.candidates)), 5)
                
                for model in self.candidates:
                    try:
                        model.fit(X_sub, y_sub, time_limit=budget_per_model)
                        preds = model.predict(X_val)
                        
                        score = float('inf')
                        if self.task in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
                            # Clip for safety
                            preds = np.clip(preds, 1e-15, 1 - 1e-15)
                            score = log_loss(y_val, preds)
                        else:
                            score = mean_squared_error(y_val, preds)
                        
                        selection_scores[model.name] = score
                    except Exception as e:
                        logger.warning(f"Selection training failed for {model.name}: {e}")
                
                # Keep Top 2
                if selection_scores:
                    best_names = sorted(selection_scores, key=selection_scores.get)[:2]
                    logger.info(f"Selected best candidates: {best_names}")
                    self.candidates = [c for c in self.candidates if c.name in best_names]
                    
            except Exception as e:
                logger.warning(f"Candidate selection failed: {e}. Proceeding with all models.")

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

        # --- Stacking (Blended Ensemble) ---
        # Construct Meta-Features
        model_names = list(val_predictions.keys())
        first_pred = val_predictions[model_names[0]]
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        
        X_meta = None
        if is_classification and first_pred.ndim == 2:
             # Multiclass: (N, C, M) -> flatten to (N, C*M) or just use probabilities as features
             X_meta = np.hstack([val_predictions[name] for name in model_names])
        else:
             # Regression/Binary(1D): (N, M)
             X_meta = np.column_stack([val_predictions[name] for name in model_names])
             
        # Train Meta-Learner
        meta_score = float('inf')
        try:
            if is_classification:
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(X_meta, y_val)
                meta_pred = self.meta_model.predict_proba(X_meta)
                if first_pred.ndim == 1: meta_pred = meta_pred[:, 1] # Binary case adjustment
                
                meta_pred = np.clip(meta_pred, 1e-15, 1 - 1e-15)
                meta_score = log_loss(y_val, meta_pred)
            else:
                self.meta_model = Ridge(alpha=1.0)
                self.meta_model.fit(X_meta, y_val)
                meta_pred = self.meta_model.predict(X_meta)
                meta_score = np.sqrt(mean_squared_error(y_val, meta_pred))
                
            logger.info(f"Stacking Score: {meta_score:.4f} vs Weighted Score: {best_score:.4f}")
            
            if meta_score < best_score:
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

        # Refit Logic Optimization
        elapsed_so_far = time.time() - start_time
        remaining_time = time_limit - elapsed_so_far
        
        # If Stacking, we need all models. If Weighted, only non-zero weights.
        if self.stacking_active:
            active_models = list(self.trained_models.keys())
        else:
            active_models = [name for name, w in self.model_weights.items() if w > 0]
        
        if active_models and remaining_time > 5:
            logger.info(f"Refitting {len(active_models)} active models on full dataset (Time left: {remaining_time:.1f}s)...")
            
            # Distribute remaining time among active models
            time_per_model = remaining_time / len(active_models)
            
            for name in active_models:
                model = self.trained_models[name]
                try:
                    # Use at least 5 seconds or the allocated share
                    budget = max(5, int(time_per_model))
                    model.fit(X, y, time_limit=budget) 
                except Exception as e:
                    logger.warning(f"Refit failed for {name}: {e}. Keeping the validation-trained version.")
        else:
            logger.info("Skipping refit due to time constraints. Using validation-trained models.")
        
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
        target_models = self.trained_models.keys() if self.stacking_active else [n for n, w in self.model_weights.items() if w > 0]

        for name in target_models:
            if name in self.trained_models:
                preds[name] = self.trained_models[name].predict(X_processed)
        
        if self.stacking_active:
            return self._stacking_predict(preds)
        else:
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
        
        # Explain only weighted ensemble for now (Stacking explainability is complex)
        # Fallback to weights if stacking is active for explanation approx
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

    def _stacking_predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        model_names = list(preds.keys()) 
        # Note: Ideally we ensure strict order match with fit. 
        # Since 'preds' iterates over 'trained_models' keys which is insertion-ordered (Python 3.7+), 
        # and 'fit' used 'val_predictions' which also followed insertion order of 'candidates', this is safe.
        
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        first_pred = next(iter(preds.values()))
        
        X_meta = None
        if is_classification and first_pred.ndim == 2:
             X_meta = np.hstack([preds[name] for name in model_names])
        else:
             X_meta = np.column_stack([preds[name] for name in model_names])
             
        if is_classification:
            final_prob = self.meta_model.predict_proba(X_meta)
            # Match output shape of base models
            if first_pred.ndim == 1: return final_prob[:, 1]
            return final_prob
        else:
            return self.meta_model.predict(X_meta)

    def _optimize_weights(self, y_true: pd.Series, preds: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], float]:
        """
        Finds optimal ensemble weights using SLSQP.
        Fixed to handle Multiclass (3D Tensor) correctly.
        """
        model_names = list(preds.keys())
        y_true_np = y_true.values
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        
        # ✅ FIX 1: จัดการ Shape ข้อมูลให้ถูกต้อง
        # ถ้าเป็น Multiclass: preds[name] คือ (N, C)
        # เราต้อง Stack เป็น (N, C, M) เพื่อให้ dot กับ weights (M,) ได้ผลลัพธ์ (N, C)
        
        try:
            # ดึง prediction array ตัวแรกมาดู shape
            first_pred = preds[model_names[0]]
            
            if is_classification and first_pred.ndim == 2:
                # Multiclass Case: Stack along last axis -> (N, Classes, Models)
                pred_tensor = np.stack([preds[name] for name in model_names], axis=-1)
            else:
                # Regression/Binary(1D): Stack columns -> (N, Models)
                pred_tensor = np.column_stack([preds[name] for name in model_names])
                
        except ValueError:
             return self._fallback_optimize_weights(y_true, preds)

        def loss_func(weights):
            # ✅ FIX 2: การคูณ Matrix ให้รองรับทั้ง 2D และ 3D
            # broadcasting weights: (M,) จะไปคูณกับ dimension สุดท้ายของ pred_tensor
            
            # ผลลัพธ์จะเป็น (N, C) สำหรับ multiclass หรือ (N,) สำหรับ regression
            final_pred = np.dot(pred_tensor, weights)
            
            if is_classification:
                # Clip เพื่อป้องกัน Log Loss ระเบิด (log(0))
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
        
        # กรอง weight ที่น้อยมากๆ ออกเพื่อลด noise
        weights_dict = {k: v for k, v in weights_dict.items() if v > 0.01}
        
        # Normalize ให้รวมกันได้ 1 เสมอ
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v/total for k, v in weights_dict.items()}
        else:
            return self._fallback_optimize_weights(y_true, preds)

        # ถ้าเป็น Regression แปลง MSE กลับเป็น RMSE เพื่อให้ดูง่าย
        if not is_classification:
            best_score = np.sqrt(best_score)

        return weights_dict, best_score

    def _fallback_optimize_weights(self, y_true, preds) -> Tuple[Dict[str, float], float]:
        """Simple inverse-error weighting if optimization fails."""
        scores = {}
        is_classification = self.task in {ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION}
        
        calculated_score = 0.0
        
        for name, pred in preds.items():
            try:
                if is_classification: 
                    # Clip ก่อนคำนวณ fallback
                    safe_pred = np.clip(pred, 1e-15, 1 - 1e-15)
                    err = log_loss(y_true, safe_pred)
                else: 
                    err = mean_squared_error(y_true, pred)
            except: 
                err = float('inf')
            
            scores[name] = 1 / (err + 1e-9)
            
            # เก็บ score ของโมเดลแรกไว้เป็นตัวแทนคร่าวๆ (ดีกว่า return 0.0)
            if calculated_score == 0.0 and err != float('inf'):
                calculated_score = err
            
        total = sum(scores.values())
        if total == 0: 
            return {k: 1.0/len(scores) for k in scores}, calculated_score
        
        weights = {k: v/total for k, v in scores.items()}
        return weights, calculated_score

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