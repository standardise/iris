import logging
from typing import Optional, Union, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings

from iris.dataset import Dataset
from iris.engine.factory import EngineFactory
from iris.foundation.types import (
    ModelBlueprint, 
    PredictionAudit, 
    Explanation,
    ProblemType
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Iris:
    """
    The main interface (Facade) for the Iris AutoML library.
    Orchestrates the Dataset, Engine, and Persistence layers.
    """
    
    def __init__(self, model_name: Optional[str] = None, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self._engine = None        
        self._blueprint: Optional[ModelBlueprint] = None
        
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO, 
                format='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            logging.basicConfig(level=logging.WARNING)

    @property
    def is_trained(self) -> bool:
        return self._engine is not None and self._blueprint is not None

    @property
    def blueprint(self) -> Optional[ModelBlueprint]:
        return self._blueprint

    def learn(self, 
              dataset: Dataset, 
              time_limit: int = 160, 
              future_steps: int = 1) -> ModelBlueprint:
        
        if self.verbose:
            logger.info(f"Starting training for task: {dataset.task_type.value}")
            logger.info(f"Configuration: time_limit={time_limit}s, future_steps={future_steps}")

        try:
            self._engine = EngineFactory.create(
                dataset.task_type, 
                future_steps=future_steps
            )
            
            self._blueprint = self._engine.fit(dataset, time_limit=time_limit)
            
            if self.verbose:
                val_score = self._blueprint.metrics.scores.get('val_score', 'N/A')
                logger.info(f"Training completed successfully. Validation Score: {val_score}")
            
            return self._blueprint

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise RuntimeError(f"Internal training error: {e}") from e

    def predict(self, 
                data: Union[pd.DataFrame, Dataset], 
                explain: bool = False) -> Union[pd.Series, PredictionAudit]:
        
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call 'learn()' first.")

        if isinstance(data, Dataset):
            df_input = data.features
        else:
            df_input = data.copy()
            if self._blueprint and self._blueprint.target_column in df_input.columns:
                df_input = df_input.drop(columns=[self._blueprint.target_column])
        
        if self.verbose:
            logger.info(f"Generating predictions for {len(df_input)} samples...")

        raw_pred = self._engine.predict(df_input)
        task = ProblemType(self._blueprint.task_type)
        
        probabilities = None
        pred_series = None

        if task in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            probabilities = raw_pred
            if raw_pred.ndim == 1:
                 pred_labels = (raw_pred > 0.5).astype(int)
            else:
                 pred_labels = np.argmax(raw_pred, axis=1)
            pred_series = pd.Series(pred_labels, index=df_input.index, name="prediction")
        else:
            pred_series = pd.Series(raw_pred, index=df_input.index, name="prediction")

        if explain:
            return self._generate_explanation(df_input, pred_series, probabilities)
        
        return pred_series

    def _generate_explanation(self, df_input, pred_series, probabilities) -> PredictionAudit:
        try:
            if self.verbose: logger.info("Calculating SHAP explanations...")
            
            base_val, contribs = self._engine.explain(df_input)
            
            prob_audit = None
            if probabilities is not None:
                if probabilities.ndim == 1:
                    prob_audit = {"class_1": float(probabilities[0])}
                elif probabilities.ndim == 2:
                    idx = 1 if probabilities.shape[1] > 1 else 0
                    prob_audit = {f"class_{idx}": float(probabilities[0][idx])}

            return PredictionAudit(
                model_version=self._blueprint.model_version,
                prediction=pred_series.iloc[0] if len(pred_series) > 0 else 0,
                probabilities=prob_audit,
                explanation=Explanation(base_value=base_val, contributions=contribs)
            )
        except Exception as e:
            logger.warning(f"Explanation calculation failed: {e}. Returning prediction only.")
            return PredictionAudit(
                model_version=self._blueprint.model_version,
                prediction=0.0,
                explanation=Explanation(base_value=0.0, contributions={"error": 0.0})
            )

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
            
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, log_loss, f1_score
        )
        
        if self.verbose: logger.info("Starting model evaluation...")

        X, y_true = dataset.get_X_y()
        raw_pred = self._engine.predict(X)
        task = ProblemType(self._blueprint.task_type)
        metrics = {}
        
        if task in (ProblemType.REGRESSION, ProblemType.TIME_SERIES_FORECASTING):
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, raw_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, raw_pred))
            metrics["r2"] = float(r2_score(y_true, raw_pred))
            
        elif task in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            try:
                safe_pred = np.clip(raw_pred, 1e-15, 1 - 1e-15)
                metrics["log_loss"] = float(log_loss(y_true, safe_pred))
            except Exception:
                metrics["log_loss"] = -1.0
            
            if raw_pred.ndim == 1:
                 pred_labels = (raw_pred > 0.5).astype(int)
            else:
                 pred_labels = np.argmax(raw_pred, axis=1)

            metrics["accuracy"] = float(accuracy_score(y_true, pred_labels))
            metrics["f1"] = float(f1_score(y_true, pred_labels, average='weighted'))
            
        if self.verbose:
            logger.info(f"Evaluation results: {metrics}")
            
        return metrics

    def save(self, path: Union[str, Path]) -> str:
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")

        path_obj = Path(path)
        if path_obj.suffix == '': 
            filename = f"{self.model_name or 'iris_model'}.joblib"
            path_obj = path_obj / filename
        
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path_obj)
            if self.verbose: logger.info(f"Model saved to: {path_obj}")
            return str(path_obj)
        except OSError as e:
            logger.error(f"Failed to save model to {path_obj}: {e}")
            raise e

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Iris":
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            loaded_instance = joblib.load(path_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to load model file: {e}") from e
        
        if not isinstance(loaded_instance, cls):
            raise TypeError(f"Loaded object is not a valid Iris model.")
            
        if hasattr(loaded_instance, 'verbose') and loaded_instance.verbose:
             logger.info(f"Model loaded from: {path}")
             
        return loaded_instance