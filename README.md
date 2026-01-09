# Iris AutoML

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Iris** is a robust, and high-performance Automated Machine Learning (AutoML) library for Python. Designed for production environments, it prioritizes stability, interpretability, and intelligent resource management over pure speed-at-all-costs.

## Key Features

- **Smart Resource Management:** Dynamically adjusts model complexity based on dataset size and time constraints.
- **Multi-Strategy Support:**
  - **Tabular:** Regression, Binary Classification, Multiclass Classification.
  - **Time Series:** Supports both Recursive (Autoregressive) and Direct (Global Regression) forecasting strategies.
- **Automated Feature Engineering:** Built-in interaction generation and target encoding.
- **Explainability:** Integrated SHAP values for model transparency.
- **Robust Architecture:** Handles categorical data and missing values gracefully without user intervention.

## Installation

```bash
pip install iris-automl
```

Or install from source:

```bash
git clone [https://github.com/standardise/iris.git](https://github.com/standardise/iris.git)
cd iris
pip install .

```

## Quick Start

### 1. Tabular Data (Regression / Classification)

Iris automatically infers the problem type based on your target variable, or you can specify it explicitly.

```python
import pandas as pd
from iris import Iris, Dataset, ProblemType

# 1. Load Data
df = pd.read_csv("insurance.csv")

# 2. Prepare Dataset
dataset = Dataset(df, target="charges")

# 3. Initialize & Train
model = Iris(model_name="insurance_regressor", verbose=True)
blueprint = model.learn(dataset, time_limit=120)

# 4. Predict
predictions = model.predict(df.sample(5))
print(predictions)

# 5. Evaluate
metrics = model.evaluate(dataset)
print(metrics)

```

### 2. Time Series Forecasting (IMPLEMENTING)

Iris supports complex time series scenarios with minimal configuration.

**Recursive Strategy (Default):** Best for short-term accuracy.

```python
dataset = Dataset(df, target="sales", date_col="date")
model = Iris(task=ProblemType.TIME_SERIES_FORECASTING)

model.learn(dataset, time_limit=60)

# Forecast next 7 days
forecast = model.predict(future_steps=7)

```

**Direct Strategy:** Best for specific date ranges or when using external features.

```python
model = Iris(task=ProblemType.TIME_SERIES_FORECASTING, strategy="direct")
model.learn(dataset)

# Forecast specific range
forecast = model.predict(start_date="2024-01-01", end_date="2024-01-31")

```

## Advanced Usage

### Model Persistence (Save / Load)

```python
# Save
model.save("models/my_model.joblib")

# Load
from iris import Iris
loaded_model = Iris.load("models/my_model.joblib")

```

### Explainability (SHAP)

Generate audit reports with feature contributions.

```python
audit = model.predict(df.iloc[0:1], explain=True)
print(f"Prediction: {audit.prediction}")
print(f"Base Value: {audit.explanation.base_value}")
print(f"Contributions: {audit.explanation.contributions}")

```

### Silent Mode

For logging-sensitive environments, run Iris in silent mode.

```python
model = Iris(verbose=False)
```

## Architecture

Iris is built on top of battle-tested libraries:

- **Core:** Scikit-Learn, NumPy, Pandas
- **Algorithms:** LightGBM, CatBoost, HistGradientBoosting, Ridge/Logistic Regression
- **Optimization:** SLSQP (Sequential Least Squares Programming) for ensemble weighting

## Requirements

- numpy>=2.1.3
- pandas>=2.3.3
- pyarrow==22.0.0
- scikit-learn>=1.7.2
- scipy>=1.16.3
- pydantic>=2.11.10
- lightgbm>=4.6.0
- catboost>=1.2.8
- shap>=0.50.0"

## License

This project is licensed under the MIT License - see the LICENSE file for details.
