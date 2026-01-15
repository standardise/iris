# Iris User Guide

This guide explains how to use Iris for different machine learning tasks, what data to send, and what rich outputs to expect.

---

## 1. Tabular Regression
**Goal:** Predict a continuous number (e.g., House Price, Temperature, Sales).

### Input Data
A standard DataFrame.
- **Target:** Numeric column.
- **Features:** Numeric, Categorical, or Date columns.

```python
from iris import Iris, Dataset
import pandas as pd

df = pd.read_csv("house_prices.csv")
# columns: ['sqft', 'bedrooms', 'city', 'price']

dataset = Dataset(src=df, target="price")
model = Iris()
model.learn(dataset)
```

### Prediction
Send a DataFrame without the target column.

```python
# Predict for a single house
new_house = pd.DataFrame({'sqft': [2000], 'bedrooms': [3], 'city': ['NY']})
result = model.predict_response(new_house)
```

### Output (`InferenceResult`)
*   **Prediction:** `540000.50` (float)
*   **Summary:** "Predicted: 540,000.50 (15.2% above average)"
*   **Visualization:** `METRIC_CARD`
    ```json
    {
      "type": "metric_card",
      "title": "Prediction vs Average",
      "data": [
        {"label": "Prediction", "value": 540000.50},
        {"label": "Global Average", "value": 460000.00}
      ]
    }
    ```

---

## 2. Tabular Classification
**Goal:** Predict a category (e.g., Spam/Ham, Iris Species, Churn).

### Input Data
- **Target:** String or Integer (Categorical).

```python
dataset = Dataset(src=df, target="species") # target values: 'setosa', 'versicolor'...
model = Iris()
model.learn(dataset)
```

### Prediction
```python
result = model.predict_response(new_flower)
```

### Output (`InferenceResult`)
*   **Prediction:** `"setosa"` (string)
*   **Summary:** "Predicted: setosa (98.5%)"
*   **Visualization:** `BAR_CHART` (Probabilities)
    ```json
    {
      "type": "bar_chart",
      "title": "Class Probabilities",
      "data": [
        {"label": "setosa", "value": 0.985},
        {"label": "versicolor", "value": 0.015},
        {"label": "virginica", "value": 0.000}
      ]
    }
    ```

---

## 3. Time Series Forecasting
**Goal:** Predict future values based on past history.

### Input Data
Requires a **Date Column**. Optionally an **ID Column** for multiple series.

```python
# columns: ['date', 'store_id', 'sales']
dataset = Dataset(
    src=df, 
    target="sales", 
    date_column="date", 
    id_column="store_id"
)
model = Iris()
model.learn(dataset)
```

### Prediction
Send a DataFrame with **Future Dates** and IDs.

```python
future = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02'],
    'store_id': ['Store_A', 'Store_A']
})
result = model.predict_response(future)
```

### Output (`InferenceResult`)
*   **Prediction:** `[100.5, 102.3]` (list of floats)
*   **Summary:** "Forecasted 2 points. Trend appears to be increasing."
*   **Visualization:** `TIME_SERIES` (History + Forecast)
    ```json
    {
      "type": "time_series",
      "title": "Historical Context & Forecast",
      "data": [
        {"x": "2023-12-30", "y": 98.0, "type": "history"},
        {"x": "2023-12-31", "y": 99.0, "type": "history"},
        {"x": "2024-01-01", "y": 100.5, "type": "forecast"},
        {"x": "2024-01-02", "y": 102.3, "type": "forecast"}
      ]
    }
    ```

---

## 4. Unsupervised Analysis
**Goal:** Discover patterns (Clusters, Anomalies) without a target.

### Clustering
```python
from iris import Analyzer
analyzer = Analyzer(task="clustering")
analyzer.fit(dataset) # No target needed

result = analyzer.predict_response(new_customer)
```
*   **Output:** "Assigned to Cluster 2" (Cluster ID).

### Anomaly Detection
```python
analyzer = Analyzer(task="anomaly")
analyzer.fit(dataset)

result = analyzer.predict_response(transaction)
```
*   **Output:** `METRIC_CARD` showing "Anomaly Score" and status "Normal" or "ANOMALY".

### Similarity Search
```python
analyzer = Analyzer(task="similarity")
analyzer.fit(dataset)

# Find 5 items similar to 'query_item'
matches = analyzer.query_similarity(query_item, k=5)
# Returns: {0: [10, 45, 99]} (Indices of similar items)
```
