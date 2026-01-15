# Iris AutoML

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Iris** is an ultra-fast, lightweight, and high-performance Automated Machine Learning (AutoML) library for Python. Powered by **Polars** and **Intel hardware acceleration**, it delivers enterprise-grade accuracy in seconds, not hours.

## Why Iris?

- **Blazing Speed:** Rust-based data processing with **Polars** and parallelized model training.
- **Hardware Optimized:** Automatically utilizes **Intel Extension for Scikit-learn** for maximum CPU performance.
- **Gap-Aware Time Series:** Robust forecasting that handles non-continuous dates, multiple IDs, and complex seasonality.
- **Smart Ensembling:** Advanced **Ridge/Logistic Stacking** meta-learner that corrects model biases automatically.
- **Enterprise Scaling:** Intelligent candidate selection for massive datasets (tested up to 400k+ rows).

## Key Features

- **Multi-Strategy Support:**
  - **Tabular:** Binary Classification, Multiclass Classification, Regression.
  - **Time Series:** Global Regression strategy with sine/cosine seasonality and trend detection.
- **Polars Feature Engineering:** Automated generation of interactions ($A 	imes B$), ratios ($A/B$), and polynomial features.
- **Explainability:** Integrated SHAP values for full model transparency.
- **Intelligent Tuning:** Adaptive hyperparameters that prevent overfitting on small data and maximize convergence on large data.

## Installation

```bash
pip install polars scikit-learn-intelex
pip install iris-automl
```

Or install from source:

```bash
git clone https://github.com/standardise/iris.git
cd iris
pip install .
```

## Quick Start & Examples

Iris simplifies the ML workflow into three steps: **Load**, **Learn**, and **Predict**.

### 1. Tabular Regression & Classification

Iris automatically detects the problem type based on your target variable.

```python
import pandas as pd
from iris import Iris, Dataset

# 1. Load Data
df = pd.read_csv("my_data.csv")

# 2. Setup Dataset
dataset = Dataset(src=df, target="target_column")

# 3. Train (Blazing fast!)
model = Iris(verbose=True)
model.learn(dataset=dataset, time_limit=60)

# 4. Predict
predictions = model.predict(new_data)
```

### 2. Multi-Series Time Series Forecasting

Iris handles multiple stores, items, or sensors in a single global model. It captures weekly/monthly seasonality and global trends automatically.

```python
from iris import Iris, Dataset

# Input: date, store_id, product_id, sales, promotion
dataset = Dataset(
    src=df,
    target="sales",
    date_column="date",
    id_column="store_id" # Groups series automatically
)

model = Iris()
model.learn(dataset, time_limit=120)

# Forecast the future by simply providing future dates and IDs
future_df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02"],
    "store_id": ["Store_A", "Store_A"]
})
forecast = model.predict(future_df)
```

## Core Technologies

### 1. The Stacking Meta-Learner

Iris doesn't just average model outputs. It trains a **Meta-Model** (Ridge or Logistic Regression) that learns _which_ base model (LGBM, CatBoost, etc.) to trust for specific data patterns. This "Stacking" strategy consistently outperforms simple weights.

### 2. High-Performance Feature Engineering

Using **Polars expressions**, Iris generates complex features like:

- **Cyclical Seasonality:** Sine and Cosine transformations of dates.
- **Contextual Aggregations:** Group-based means and standard deviations.
- **Non-Linear terms:** Polynomials and safe Ratios.

### 3. Dynamic Budgeting

The **Smart Refit** logic ensures Iris never exceeds your `time_limit`. It calculates the remaining time after validation and intelligently decides whether to retrain on the full dataset or fallback to validation models.

## Requirements

- **polars** >= 1.0.0
- **scikit-learn-intelex** >= 2024.0
- **numpy** >= 2.1.3
- **pandas** >= 2.3.3
- **scikit-learn** >= 1.7.2
- **lightgbm** >= 4.6.0
- **catboost** >= 1.2.8
- **shap** >= 0.50.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.
