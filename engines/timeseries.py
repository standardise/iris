from iris.engines.base import BaseEngine
from iris.core.types import ProblemType, ModelBlueprint, FeatureSchema, ModelMetrics
from iris.core.types import InferenceResult, VisualizationData, VisualizationType
from iris.models.supervised import LGBMRegressorModel, CatBoostRegressorModel, RidgeRegressorModel
from iris.features.timeseries import TimeSeriesFeatureEngineer
import pandas as pd
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class TimeSeriesEngine(BaseEngine):
    def __init__(self, date_col: str, id_col: str = None):
        super().__init__(task=ProblemType.TIME_SERIES_FORECASTING)
        self.date_col = date_col
        self.id_col = id_col
        self.ts_engineer = None
        self.history = None # Store tail of training data for plots

    def fit(self, dataset, time_limit=300):
        # ... (Feature Engineering) ...
        logger.info("Starting Time Series Feature Engineering...")
        self.ts_engineer = TimeSeriesFeatureEngineer(self.date_col, self.id_col)
        
        df_processed = self.ts_engineer.fit_transform(dataset.df)
        
        # Store History (Last 60 points) for Visualization
        # We need date and target.
        if self.id_col:
            # If multi-series, store history for each ID? Too heavy.
            # Just store the global aggregate or the last few rows raw.
            # Let's store the last 100 rows raw to be safe.
            self.history = dataset.df.iloc[-100:][[self.date_col, dataset.target_name, self.id_col]].copy()
        else:
            self.history = dataset.df.iloc[-60:][[self.date_col, dataset.target_name]].copy()
            
        # ... (Rest of fit logic) ...
        
        X = df_processed.drop(columns=[self.date_col])
        if dataset.target_name in X.columns:
            X = X.drop(columns=[dataset.target_name])
            
        y = dataset.target.reset_index(drop=True)
        
        # Ensure consistent feature order for strict models like CatBoost
        X = X.reindex(sorted(X.columns), axis=1)
        
        # 2. Register Time Series Models
        self.candidates = []
        self.register_candidates([
            CatBoostRegressorModel("CatBoost_TS_Fast", mode="fast"),
            CatBoostRegressorModel("CatBoost_TS_Accurate", mode="accurate"),
            LGBMRegressorModel("LGBM_TS", mode="accurate"),
            RidgeRegressorModel("Linear_Trend") 
        ])
        
        # 3. Custom Train/Val Split
        full_data = pd.concat([X, y], axis=1).sort_values(by="time_idx")
        n_samples = len(full_data)
        train_size = int(n_samples * 0.85)
        
        train_data = full_data.iloc[:train_size]
        val_data = full_data.iloc[train_size:]
        
        target_name = dataset.target_name
        if y.name is None: y.name = target_name
        
        X_train = train_data.drop(columns=[y.name])
        y_train = train_data[y.name]
        X_val = val_data.drop(columns=[y.name])
        y_val = val_data[y.name]
        
        self.feature_dtypes = X.dtypes.to_dict()
        
        logger.info(f"TRAIN COLS: {list(X_train.columns)}")
        logger.info(f"TS Engine started. Train: {len(X_train)}, Val: {len(X_val)}")
        
        start_time = time.time()
        
        # Call shared fit logic
        best_score = self._fit_candidates_and_ensemble(X_train, y_train, X_val, y_val, time_limit, start_time)

        # Refit Logic
        elapsed_so_far = time.time() - start_time
        remaining_time = time_limit - elapsed_so_far
        
        target_models = self.trained_models.keys() if self.stacking_active else [n for n, w in self.model_weights.items() if w > 0]
        
        if remaining_time > 5:
            logger.info("Refitting best models on full history...")
            budget = max(5, int(remaining_time / len(target_models)))
            for name in target_models:
                try:
                    self.trained_models[name].fit(X, y, time_limit=budget)
                except Exception as e:
                    logger.warning(f"Refit failed for {name}: {e}")

        return self._create_blueprint(dataset, best_score)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Preprocess
        df_processed = self.ts_engineer.transform(X)
        if self.date_col in df_processed.columns:
            df_processed = df_processed.drop(columns=[self.date_col])
            
        # Ensure consistent feature order
        df_processed = df_processed.reindex(sorted(df_processed.columns), axis=1)
            
        return super().predict(df_processed)

    def predict_response(self, dataset_X: pd.DataFrame) -> InferenceResult:
        """
        Generates a rich prediction response with visualization data.
        dataset_X: The dataframe containing future dates (and IDs).
        """
        # 1. Raw Prediction
        preds = self.predict(dataset_X)
        
        # 2. Build Summary
        mean_pred = float(np.mean(preds))
        trend = "stable"
        if self.history is not None:
            last_val = self.history.iloc[-1, 1] # Assumes col 1 is target
            if mean_pred > last_val * 1.05: trend = "increasing"
            elif mean_pred < last_val * 0.95: trend = "decreasing"
            
        summary = f"Forecasted {len(preds)} points. Trend appears to be {trend} (Avg: {mean_pred:.2f})."
        
        # 3. Build Visualization Data (History + Forecast)
        plot_data = []
        
        # Add History
        if self.history is not None:
            hist_df = self.history.copy()
            # Rename cols for consistency: x=date, y=value, series=id
            hist_df.columns = ['date', 'value'] + (['series'] if self.id_col else [])
            for _, row in hist_df.iterrows():
                pt = {"x": str(row['date']), "y": float(row['value']), "type": "history"}
                if self.id_col: pt["series"] = str(row['series'])
                plot_data.append(pt)
                
        # Add Forecast
        # dataset_X has dates
        dates = dataset_X[self.date_col].astype(str).values
        ids = dataset_X[self.id_col].astype(str).values if self.id_col else [None]*len(dates)
        
        for dt, val, ser in zip(dates, preds, ids):
            pt = {"x": dt, "y": float(val), "type": "forecast"}
            if ser: pt["series"] = ser
            plot_data.append(pt)
            
        viz = VisualizationData(
            type=VisualizationType.TIME_SERIES,
            title="Historical Context & Forecast",
            data=plot_data,
            axes={"x": "Date", "y": "Value"}
        )
        
        return InferenceResult(
            prediction=preds.tolist(),
            summary=summary,
            details={"model_count": len(self.model_weights), "horizon": len(preds)},
            visualization=viz
        )