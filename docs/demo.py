import pandas as pd
import numpy as np
from iris import Iris, Dataset, Analyzer

def demo_regression():
    print("\n=== 1. DEMO: Tabular Regression (House Prices) ===")
    
    # 1. Create Dummy Data
    data = {
        'sqft': [1000, 1500, 2000, 2500, 3000, 1200, 1800, 2200],
        'bedrooms': [2, 3, 3, 4, 5, 2, 3, 4],
        'price': [200, 300, 400, 500, 600, 250, 350, 450]
    }
    df = pd.DataFrame(data)
    
    # 2. Setup Dataset
    # Iris automatically detects this is a Regression task because 'price' is numeric
    ds = Dataset(src=df, target="price")
    
    # 3. Train
    model = Iris(model_name="house_pricer", verbose=False)
    model.learn(ds, time_limit=10) # Fast training
    
    # 4. Predict with Rich Output
    new_house = pd.DataFrame({'sqft': [2100], 'bedrooms': [3]})
    result = model.predict_response(new_house)
    
    print(f"Summary: {result.summary}")
    print(f"Raw Prediction: {result.prediction}")
    print(f"Visualization Data: {result.visualization.model_dump_json(indent=2)}")


def demo_timeseries():
    print("\n=== 2. DEMO: Time Series Forecasting (Sales) ===")
    
    # 1. Create Dummy Time Series (2 Stores)
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    
    # Store A (Trend Up)
    df_a = pd.DataFrame({
        'date': dates,
        'store_id': 'Store_A',
        'sales': np.linspace(100, 200, 60) + np.random.normal(0, 5, 60)
    })
    
    # Store B (Flat)
    df_b = pd.DataFrame({
        'date': dates,
        'store_id': 'Store_B',
        'sales': np.linspace(50, 50, 60) + np.random.normal(0, 2, 60)
    })
    
    df = pd.concat([df_a, df_b])
    
    # 2. Setup Dataset
    ds = Dataset(
        src=df, 
        target="sales", 
        date_column="date", 
        id_column="store_id"
    )
    
    # 3. Train
    model = Iris(model_name="sales_forecaster", verbose=False)
    model.learn(ds, time_limit=15)
    
    # 4. Forecast Next 3 Days
    future = pd.DataFrame({
        'date': ["2024-03-02", "2024-03-03", "2024-03-04"] * 2,
        'store_id': ["Store_A"]*3 + ["Store_B"]*3
    })
    
    result = model.predict_response(future)
    
    print(f"Summary: {result.summary}")
    # The visualization contains the history + forecast points ready for plotting
    print(f"Plot Data Points: {len(result.visualization.data)}")
    print(f"First 2 Plot Points: {result.visualization.data[:2]}")

def demo_clustering():
    print("\n=== 3. DEMO: Unsupervised (Customer Clusters) ===")
    
    # 1. Create Data (No Target)
    df = pd.DataFrame({
        'spend': [100, 120, 110, 500, 550, 520],
        'visits': [1, 2, 1, 10, 12, 11]
    })
    
    ds = Dataset(src=df, target=None) # No target needed
    
    # 2. Analyze
    analyzer = Analyzer(task="clustering", verbose=False)
    analyzer.fit(ds, n_clusters=2)
    
    # 3. Get Results
    new_customer = pd.DataFrame({'spend': [115], 'visits': [2]})
    result = analyzer.predict_response(new_customer)
    
    print(f"Summary: {result.summary}")

if __name__ == "__main__":
    demo_regression()
    demo_timeseries()
    demo_clustering()
