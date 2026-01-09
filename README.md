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
git clone https://github.com/standardise/iris.git
cd iris
pip install .
```

## Quick Start & Examples

Iris simplifies the ML workflow into three steps: **Load**, **Learn**, and **Predict**. It automatically detects the problem type (Regression, Binary, or Multiclass) based on your target variable.

### 1. Regression (Insurance Dataset)

_Predicting continuous values (e.g., costs, prices)._

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from iris import Iris, Dataset

# 1. Load Data
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 2. Setup Dataset
train_ds = Dataset(src=train, target="charges")
test_ds = Dataset(src=test, target="charges")

# 3. Train
model = Iris(verbose=True)
model.learn(dataset=train_ds, time_limit=60)

# 4. Predict
# Returns continuous values
predictions = model.predict(test_ds)
print(predictions.head())

# 5. Evaluate
metrics = model.evaluate(test_ds)
print(f"RMSE: {metrics['rmse']:.2f}")
```

### 2. Binary Classification (Titanic Dataset)

_Predicting one of two classes (e.g., Yes/No, 0/1)._

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from iris import Iris, Dataset

# 1. Load Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']) # Optional cleanup
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 2. Train
train_ds = Dataset(src=train, target="Survived")
model = Iris(verbose=True)
model.learn(dataset=train_ds, time_limit=60)

# 3. Predict Classes (returns 0 or 1)
preds = model.predict(test)
print("Predictions:", preds.head())

# 4. Predict Probabilities (returns DataFrame with columns [0, 1])
probs = model.predict_proba(test)
print("Probabilities:
", probs.head())
```

**Output:**

```text
Predictions:
709    0
439    0
840    0
Name: prediction, dtype: int64

Probabilities:
          0         1
709  0.799335  0.200665
439  0.755787  0.244213
```

### 3. Multiclass Classification (Iris Dataset)

_Predicting one of many classes (e.g., setosa, versicolor, virginica)._

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from iris import Iris, Dataset

# 1. Load Data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])

# 2. Train
train_ds = Dataset(src=train, target="species")
model = Iris(verbose=True)
model.learn(dataset=train_ds, time_limit=60)

# 3. Predict Classes (returns strings: 'setosa', 'virginica'...))
preds = model.predict(test)

# 4. Predict Probabilities (returns DataFrame with columns for each class)
probs = model.predict_proba(test)
print("Probabilities:
", probs.head())

# 5. Evaluate
test_ds = Dataset(src=test, target="species")
metrics = model.evaluate(test_ds)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

**Output:**

```text
Probabilities:
       setosa  versicolor  virginica
38   0.978483    0.015262   0.006255
127  0.020856    0.198125   0.781020
57   0.068282    0.908386   0.023332
```

## 📖 Prediction Methods Explained

| Method                      | Returns        | Description                                                                                                                                                                   |
| :-------------------------- | :------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.predict(data)`       | `pd.Series`    | **Best Guess.** <br> • **Regression:** The predicted value.<br> • **Classification:** The predicted class label (e.g., "spam", 0).                                            |
| `model.predict_proba(data)` | `pd.DataFrame` | **Confidence Scores.** <br> • **Regression:** _Not available._<br> • **Classification:** A DataFrame where columns are class labels and values are probabilities (0.0 - 1.0). |

## Time Series Forecasting

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

## Requirements

- numpy>=2.1.3
- pandas>=2.3.3
- pyarrow==2.0.0
- scikit-learn>=1.7.2
- scipy>=1.16.3
- pydantic>=2.11.10
- lightgbm>=4.6.0
- catboost>=1.2.8
- shap>=0.50.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.
