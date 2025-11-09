"""
Script to create and save a scaler for the model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def create_scaler():
    """Create and save a scaler based on the dataset"""
    
    data_path = Path("cleaned_aqi_weather_dataset.csv")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Process date columns
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['aqi_index', 'Calculated_AQI'])
    
    # Define features and target - match exactly how model was trained
    # Model training: X = df.drop(columns=target_columns).select_dtypes(exclude=["datetime64[ns]"])
    target_columns = ['aqi_index', 'Calculated_AQI']
    
    # Drop target columns, then select only numeric (non-datetime) columns
    # This includes 'index' column to match model training (23 features)
    all_features = [col for col in df.columns if col not in target_columns]
    X = df[all_features].select_dtypes(exclude=["datetime64[ns]"])
    
    # Create and fit scaler
    print("Creating scaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save scaler
    scaler_path = Path("scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f" Scaler saved to {scaler_path}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Samples: {len(X)}")

if __name__ == "__main__":
    create_scaler()

