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
  - **Analysis:** Unsupervised Clustering, Anomaly Detection, and Similarity Search.
- **Polars Feature Engineering:** Automated generation of interactions ($A \times B$), ratios ($A/B$), and polynomial features.
- **Contextual Data Output:** Built-in support for rich inference results with metadata for plotting.
- **Explainability:** Integrated SHAP values for full model transparency.

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

```python
from iris import Iris, Dataset

dataset = Dataset(src=df, target="target_column")
model = Iris()
model.learn(dataset, time_limit=60)

# Use predict_response for rich summaries and data context
result = model.predict_response(new_data)
print(result.summary)
```

### 2. Multi-Series Time Series Forecasting

```python
from iris import Iris, Dataset

dataset = Dataset(src=df, target="sales", date_column="date", id_column="store_id")
model = Iris()
model.learn(dataset)

# Context includes historical points + future forecast for easy plotting
result = model.predict_response(future_df)
```

### 3. Unsupervised Analysis (Clustering & Anomaly)

```python
from iris import Analyzer, Dataset

analyzer = Analyzer(task="clustering")
analyzer.fit(dataset, n_clusters=5)
df_clustered = analyzer.get_clusters(dataset)
```

## Contextual Data Format

The `predict_response()` method returns an `InferenceResult` object designed for easy integration with frontend charts (e.g. Recharts, Chart.js):

| Field | Type | Description |
| :--- | :--- | :--- |
| `prediction` | `Any` | The raw result (value, class, or cluster ID). |
| `summary` | `str` | A human-readable summary (e.g. "Predicted 15% above average"). |
| `context` | `Object` | JSON-friendly data for charts (Time Series, Distribution, or Metric). |
| `details` | `dict` | Advanced metrics and model metadata. |

## Core Technologies

### 1. The Stacking Meta-Learner
Iris doesn't just average model outputs. It trains a **Meta-Model** (Ridge or Logistic Regression) that learns *which* base model (LGBM, CatBoost, etc.) to trust for specific data patterns.

### 2. High-Performance Feature Engineering
Using **Polars expressions**, Iris generates complex features like Cyclical Seasonality, Contextual Aggregations, and Non-Linear terms at high speed.

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
