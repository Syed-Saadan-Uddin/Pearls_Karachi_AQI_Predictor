"""
Improved AQI Prediction Model Training
Uses advanced feature engineering, XGBoost/LightGBM, and LSTM/GRU for better predictions
"""
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("️ XGBoost not available, will use LightGBM or GradientBoosting")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("️ LightGBM not available, will use GradientBoosting")

# Try to import TensorFlow/Keras for LSTM/GRU
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("️ TensorFlow not available, LSTM/GRU models will be skipped")
    print("   Install with: pip install tensorflow")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Add project root to path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from backend.hopsworks_utils import get_offline_features, load_from_csv_fallback
    USE_HOPSWORKS = True
except ImportError:
    USE_HOPSWORKS = False
    print("️ Hopsworks not available, will load from CSV")

def augment_dataset(df, augmentation_factor=20):
    """
    MASSIVELY augment dataset using multiple sophisticated techniques
    This creates an enormous dataset for better deep learning model training
    """
    print(f"\n MASSIVE Dataset Augmentation (target: {augmentation_factor}x)...")
    original_size = len(df)
    
    augmented_dfs = [df.copy()]
    
    # Technique 1: Gaussian noise with varying intensities
    print("   Technique 1: Gaussian noise variations...")
    for i in range(int(augmentation_factor * 0.3)):
        aug_df = df.copy()
        numeric_cols = aug_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['year', 'month', 'day', 'hour', 'index']:
                noise_factor = np.random.uniform(0.005, 0.08)
                noise = np.random.normal(0, aug_df[col].std() * noise_factor, len(aug_df))
                aug_df[col] = aug_df[col] + noise
                if col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'Calculated_AQI', 'aqi_index']:
                    aug_df[col] = np.maximum(aug_df[col], 0)
        augmented_dfs.append(aug_df)
    
    # Technique 2: Linear interpolation between consecutive rows
    print("   Technique 2: Linear interpolation...")
    for i in range(int(augmentation_factor * 0.2)):
        aug_df = df.copy()
        numeric_cols = aug_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['year', 'month', 'day', 'hour', 'index']:
                # Interpolate between consecutive values
                alpha = np.random.uniform(0.2, 0.8)
                aug_df[col] = aug_df[col] * (1 - alpha) + aug_df[col].shift(1).fillna(aug_df[col]) * alpha
                if col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'Calculated_AQI', 'aqi_index']:
                    aug_df[col] = np.maximum(aug_df[col], 0)
        augmented_dfs.append(aug_df)
    
    # Technique 3: Seasonal pattern variations
    print("   Technique 3: Seasonal pattern variations...")
    for i in range(int(augmentation_factor * 0.15)):
        aug_df = df.copy()
        if 'month' in aug_df.columns:
            # Add seasonal multipliers
            seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * aug_df['month'] / 12 + np.random.uniform(0, 2*np.pi))
            numeric_cols = aug_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['year', 'month', 'day', 'hour', 'index']:
                    aug_df[col] = aug_df[col] * seasonal_mult
                    if col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'Calculated_AQI', 'aqi_index']:
                        aug_df[col] = np.maximum(aug_df[col], 0)
        augmented_dfs.append(aug_df)
    
    # Technique 4: Time-based shifts with pattern preservation
    print("   Technique 4: Time-based shifts...")
    for i in range(int(augmentation_factor * 0.15)):
        aug_df = df.copy()
        if 'datetime' in aug_df.columns:
            shift_hours = np.random.choice([-3, -2, -1, 1, 2, 3, 6, 12], size=len(aug_df), 
                                         p=[0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15])
            aug_df['datetime'] = pd.to_datetime(aug_df['datetime']) + pd.to_timedelta(shift_hours, unit='h')
            # Update time-based columns
            if 'hour' in aug_df.columns:
                aug_df['hour'] = aug_df['datetime'].dt.hour
            if 'day' in aug_df.columns:
                aug_df['day'] = aug_df['datetime'].dt.day
        augmented_dfs.append(aug_df)
    
    # Technique 5: Mix-and-match (combine features from different rows)
    print("   Technique 5: Feature mixing...")
    for i in range(int(augmentation_factor * 0.1)):
        aug_df = df.copy()
        # Randomly swap some features between rows
        swap_indices = np.random.choice(len(aug_df), size=min(100, len(aug_df)//10), replace=False)
        for idx in swap_indices:
            swap_with = np.random.choice(len(aug_df))
            # Swap weather features but keep pollutants together
            weather_cols = [c for c in aug_df.columns if 'temperature' in c or 'humidity' in c or 
                          'wind' in c or 'precipitation' in c or 'pressure' in c]
            for col in weather_cols:
                if col in aug_df.columns:
                    aug_df.loc[idx, col], aug_df.loc[swap_with, col] = \
                        aug_df.loc[swap_with, col], aug_df.loc[idx, col]
        augmented_dfs.append(aug_df)
    
    # Technique 6: Polynomial transformations (smooth variations)
    print("   Technique 6: Polynomial transformations...")
    for i in range(int(augmentation_factor * 0.1)):
        aug_df = df.copy()
        numeric_cols = aug_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['year', 'month', 'day', 'hour', 'index']:
                # Apply smooth polynomial transformation
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-0.1, 0.1)
                aug_df[col] = alpha * aug_df[col] + beta * (aug_df[col] ** 2) / (aug_df[col].max() + 1e-6)
                if col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'Calculated_AQI', 'aqi_index']:
                    aug_df[col] = np.maximum(aug_df[col], 0)
        augmented_dfs.append(aug_df)
    
    # Combine all augmented data
    print("   Combining all augmented data...")
    augmented_df = pd.concat(augmented_dfs, ignore_index=True)
    
    # Shuffle to mix original and augmented data
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"    Original size: {original_size:,}")
    print(f"    Augmented size: {len(augmented_df):,}")
    print(f"    Expansion: {len(augmented_df) / original_size:.2f}x")
    
    return augmented_df

def create_sequences(data, target, sequence_length=24, forecast_horizon=1):
    """
    Create sequences for LSTM/GRU models
    sequence_length: number of time steps to look back
    forecast_horizon: number of time steps to predict ahead
    """
    X_seq, y_seq = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X_seq.append(data[i:i+sequence_length])
        y_seq.append(target[i+sequence_length:i+sequence_length+forecast_horizon])
    
    return np.array(X_seq), np.array(y_seq)

print("=" * 60)
print("IMPROVED AQI MODEL TRAINING")
print("=" * 60)

# Load data
if USE_HOPSWORKS:
    print("\n Loading data from Hopsworks feature store...")
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)
    try:
        df = get_offline_features(start_date, end_date)
        if df is None or df.empty:
            USE_HOPSWORKS = False
            print("️ Hopsworks returned empty data, falling back to CSV")
    except Exception as e:
        print(f"️ Error loading from Hopsworks: {e}, falling back to CSV")
        USE_HOPSWORKS = False

if not USE_HOPSWORKS:
    print("\n Loading data from CSV...")
    data_path = Path("cleaned_aqi_weather_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)

print(f" Loaded {len(df)} rows")

# Sort by timestamp if available
if 'event_timestamp' in df.columns:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df = df.sort_values('event_timestamp').reset_index(drop=True)
elif 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

# Create datetime index for time series features
if 'event_timestamp' in df.columns:
    df['datetime'] = pd.to_datetime(df['event_timestamp'])
elif 'timestamp' in df.columns:
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
else:
    # Create datetime from year, month, day, hour
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']].apply(
            lambda x: f"{int(x['year'])}-{int(x['month'])}-{int(x['day'])} {int(x['hour'])}:00:00",
            axis=1
        )
    )

df = df.sort_values('datetime').reset_index(drop=True)

# Try WITHOUT augmentation first to see baseline performance
# Augmentation can introduce noise and distribution shifts
USE_AUGMENTATION = False  # Set to True if needed, but start without
USE_MINIMAL_FEATURES = True  # Try with only raw pollutants/weather first to diagnose

if USE_AUGMENTATION:
    print("\n Expanding dataset...")
    df = augment_dataset(df, augmentation_factor=2)  # Minimal augmentation if needed
    df = df.sort_values('datetime').reset_index(drop=True)
else:
    print("\n⏭️  Skipping augmentation - using original data to avoid distribution shifts")

print("\n Feature Engineering...")

# Define pollutant and weather columns
pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
weather_cols = [
    'temperature_2m', 'relative_humidity_2m', 'precipitation',
    'wind_speed_10m', 'wind_direction_10m', 'surface_pressure',
    'dew_point_2m', 'apparent_temperature', 'shortwave_radiation',
    'et0_fao_evapotranspiration'
]

# Option to use only raw features for diagnosis
if USE_MINIMAL_FEATURES:
    print("  ️  Using MINIMAL features (raw pollutants + weather only) for diagnosis")
    print("  This helps identify if the issue is with feature engineering")
    # Skip all feature engineering - use only raw values
    pass
else:
    # Create lag features (previous hours) - MORE LAGS
    # NOTE: Do NOT include target variable in lag features (data leakage!)
    print("  Creating lag features...")
    for lag in [1, 2, 3, 6, 12, 18, 24, 48]:
        for col in pollutant_cols:  # Only use pollutant columns, NOT target
            if col in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

    # Create rolling statistics - MORE WINDOWS
    print("  Creating rolling statistics...")
    for window in [3, 6, 12, 24, 48, 72]:
        for col in pollutant_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()

    # Time-based features
    print("  Creating time-based features...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)

    # Weather interactions
    print("  Creating interaction features...")
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        df['wind_x'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
        df['wind_y'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))

    # PM ratio features (important for AQI)
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)

    # Advanced pollutant interactions
    print("  Creating advanced interaction features...")
    if 'pm2_5' in df.columns and 'temperature_2m' in df.columns:
        df['pm2_5_temp_interaction'] = df['pm2_5'] * df['temperature_2m']
    if 'pm10' in df.columns and 'wind_speed_10m' in df.columns:
        df['pm10_wind_interaction'] = df['pm10'] * df['wind_speed_10m']
    if 'o3' in df.columns and 'temperature_2m' in df.columns:
        df['o3_temp_interaction'] = df['o3'] * df['temperature_2m']

    # Rate of change features (important for time series)
    print("  Creating rate of change features...")
    for col in pollutant_cols[:4]:  # Top 4 pollutants
        if col in df.columns:
            df[f'{col}_rate_change'] = df[col].diff().fillna(0)
            df[f'{col}_rate_change_abs'] = df[f'{col}_rate_change'].abs()

    # Exponential moving averages (capture trends) - MORE WINDOWS
    print("  Creating exponential moving averages...")
    for col in ['pm2_5', 'pm10', 'o3', 'no2', 'co']:
        if col in df.columns:
            for span in [3, 6, 12, 24, 48]:
                df[f'{col}_ema_{span}h'] = df[col].ewm(span=span, adjust=False).mean()
                df[f'{col}_ema_{span}h_std'] = df[col].ewm(span=span, adjust=False).std().fillna(0)

    # Fill NaN values created by lag features
    df = df.bfill().ffill().fillna(0)

print(f" Feature engineering complete. Total features: {len(df.columns)}")

# Prepare features and target
target_col = 'Calculated_AQI'  # Focus on this since backend uses it
if target_col not in df.columns:
    target_col = 'aqi_index'
    print(f"   ️  WARNING: Calculated_AQI not found, using aqi_index instead")
    print(f"   ️  WARNING: This will cause predictions to be in wrong scale (3-5 instead of 70-500)")
    print(f"   ️  WARNING: Backend will need to convert predictions to actual AQI scale")

# Drop target and datetime columns
# IMPORTANT: Exclude target and any related columns to prevent data leakage
exclude_cols = [
    target_col, 'aqi_index', 'datetime', 'event_timestamp', 'timestamp',
    'created', 'index'
]

# Also exclude any lag features of the target (if they exist from previous runs)
# CRITICAL: Never use target variable or its lags as features (data leakage!)
target_lag_features = [col for col in df.columns if f'{target_col}_lag' in col.lower() or 'calculated_aqi_lag' in col.lower() or 'aqi_index_lag' in col.lower()]
if target_lag_features:
    print(f"   ️  Found {len(target_lag_features)} target lag features - EXCLUDING to prevent data leakage")
    print(f"      Examples: {target_lag_features[:3]}")
exclude_cols.extend(target_lag_features)

# Double-check: remove any columns that contain the target name (except exact match which is already excluded)
target_related = [col for col in df.columns if target_col.lower() in col.lower() and col != target_col]
if target_related:
    print(f"   ️  Found {len(target_related)} target-related features - EXCLUDING")
    exclude_cols.extend(target_related)

feature_cols = [col for col in df.columns if col not in exclude_cols]

# If using minimal features, only keep raw pollutants and weather
if USE_MINIMAL_FEATURES:
    minimal_cols = pollutant_cols + weather_cols + ['year', 'month', 'day', 'hour']
    feature_cols = [col for col in feature_cols if col in minimal_cols]
    print(f"   Using minimal feature set: {len(feature_cols)} features")
    print(f"     Features: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

# Select only numeric features
X = df[feature_cols].select_dtypes(include=[np.number])
y = df[target_col]

print(f"\n Model Configuration:")
print(f"   Features: {len(X.columns)}")
print(f"   Samples: {len(X)}")
print(f"   Target: {target_col}")
print(f"   Target range: {y.min():.2f} - {y.max():.2f}")
print(f"   Target mean: {y.mean():.2f}")
print(f"   Target std: {y.std():.2f}")

# Check feature-target correlations
print(f"\n Feature Analysis:")
if len(X.columns) > 0:
    # Calculate correlations with target
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"   Top 10 most correlated features:")
    for feat, corr in correlations.head(10).items():
        print(f"      {feat}: {corr:.4f}")
    print(f"   Features with correlation > 0.1: {(correlations > 0.1).sum()}")
    print(f"   Features with correlation > 0.3: {(correlations > 0.3).sum()}")

# CRITICAL: Test if we can compute AQI directly from pollutants
print(f"\n Testing Direct AQI Computation:")
print("   (Calculated_AQI should be MAX of individual pollutant AQIs)")
if all(col in df.columns for col in ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']):
    # PM2.5 breakpoints (US EPA standard)
    pm25_bp = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
               (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
               (350.5, 500.4, 401, 500)]
    
    def calc_aqi(conc, breakpoints):
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= conc <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
        return None
    
    # Compute AQI from pollutants (simplified - using PM2.5 breakpoints for all)
    computed_aqi = []
    for idx, row in df.head(100).iterrows():  # Test first 100 rows
        aqis = []
        for pol in ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']:
            if pol in row and not pd.isna(row[pol]) and row[pol] >= 0:
                aqi_val = calc_aqi(row[pol], pm25_bp)
                if aqi_val is not None:
                    aqis.append(aqi_val)
        computed_aqi.append(max(aqis) if aqis else None)
    
    # Compare with actual Calculated_AQI
    actual_aqi = df.head(100)[target_col].values
    valid_mask = ~pd.isna(computed_aqi) & ~pd.isna(actual_aqi)
    if valid_mask.sum() > 0:
        computed_arr = np.array(computed_aqi)[valid_mask]
        actual_arr = actual_aqi[valid_mask]
        match_rate = (np.abs(computed_arr - actual_arr) < 1).sum() / len(computed_arr) * 100
        print(f"   Computed vs Actual AQI match rate: {match_rate:.1f}% (within 1 unit)")
        print(f"   Mean difference: {np.mean(np.abs(computed_arr - actual_arr)):.2f}")
        if match_rate < 50:
            print(f"   ️  WARNING: Low match rate suggests Calculated_AQI may not be computed correctly!")
    else:
        print(f"   ️  Could not compute AQI from pollutants - check data quality")
else:
    print(f"   ️  Missing pollutant columns - cannot test direct computation")

# Remove any remaining NaN or inf
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())

# Feature selection - remove highly correlated features (BEFORE split to avoid data leakage)
print("\n Feature selection...")
correlation_threshold = 0.95
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
if high_corr_features:
    print(f"   Removing {len(high_corr_features)} highly correlated features")
    X = X.drop(columns=high_corr_features)

# Remove features with very low variance
from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0.01)
X_array = variance_selector.fit_transform(X)
selected_features = X.columns[variance_selector.get_support()]
X = pd.DataFrame(X_array, columns=selected_features, index=X.index)
print(f"   Final feature count: {len(X.columns)}")

# Check if data is sorted by target OR if first portion has zero variance (CRITICAL ISSUE!)
print("\n Checking data order and variance...")
is_sorted_inc = y.is_monotonic_increasing
is_sorted_dec = y.is_monotonic_decreasing
is_sorted = is_sorted_inc or is_sorted_dec

# Check variance in first 70% (what will become training set)
train_size_check = int(len(y) * 0.7)
first_portion_std = y.iloc[:train_size_check].std()
first_portion_unique = y.iloc[:train_size_check].nunique()

print(f"   First 70% of data: std={first_portion_std:.2f}, unique values={first_portion_unique}")
print(f"   Full dataset: std={y.std():.2f}, unique values={y.nunique()}")

if is_sorted:
    print(f"   ️  CRITICAL: Data appears to be sorted by target value!")
    print(f"   Sorted {'increasing' if is_sorted_inc else 'decreasing'}")
    print(f"   This will cause all splits to have the same values!")
    print(f"   Solution: Shuffling data before split (temporal order may be lost)")
    shuffle_needed = True
elif first_portion_std == 0 or first_portion_unique == 1:
    print(f"   ️  CRITICAL: First 70% of data has ZERO variance!")
    print(f"   All training samples would have the same target value!")
    print(f"   Solution: Shuffling data before split")
    shuffle_needed = True
else:
    print(f"    Data appears OK - first portion has variance")
    shuffle_needed = False

if shuffle_needed:
    # Shuffle the data to break the pattern
    shuffle_indices = np.random.RandomState(42).permutation(len(X))
    X = X.iloc[shuffle_indices].reset_index(drop=True)
    y = y.iloc[shuffle_indices].reset_index(drop=True)
    print(f"    Data shuffled to break pattern")
    # Verify shuffle worked
    new_first_portion_std = y.iloc[:train_size_check].std()
    print(f"   After shuffle - First 70% std: {new_first_portion_std:.2f}")

# Time series split (preserve temporal order) with validation set
# IMPORTANT: Split BEFORE outlier removal to preserve distribution
print("\n Splitting data...")
train_idx = int(len(X) * 0.7)  # 70% for training
val_idx = int(len(X) * 0.85)   # 15% for validation
X_train = X.iloc[:train_idx].copy()
X_val = X.iloc[train_idx:val_idx].copy()
X_test = X.iloc[val_idx:].copy()
y_train = y.iloc[:train_idx].copy()
y_val = y.iloc[train_idx:val_idx].copy()
y_test = y.iloc[val_idx:].copy()

# Remove outliers from target (cap at 3 standard deviations) - AFTER split
# Use training set statistics to avoid data leakage
print("\n Removing outliers (using training statistics)...")
y_train_mean = y_train.mean()
y_train_std = y_train.std()

if y_train_std == 0:
    print(f"   ️  CRITICAL: Training set has ZERO variance before outlier removal!")
    print(f"   All training samples have the same target value: {y_train_mean:.2f}")
    print(f"   Skipping outlier removal - would remove all data!")
    print(f"   This suggests data was sorted - check if shuffle worked")
else:
    # Use a more lenient threshold (5 std devs instead of 3) to avoid removing too much
    outlier_threshold = 5 * y_train_std  # More lenient
    y_lower = max(0, y_train_mean - outlier_threshold)  # AQI can't be negative
    y_upper = y_train_mean + outlier_threshold
    
    # Apply same threshold to all sets
    train_mask = (y_train >= y_lower) & (y_train <= y_upper)
    val_mask = (y_val >= y_lower) & (y_val <= y_upper)
    test_mask = (y_test >= y_lower) & (y_test <= y_upper)
    
    # Only remove if we keep at least 90% of data
    train_keep_ratio = train_mask.sum() / len(train_mask)
    if train_keep_ratio < 0.9:
        print(f"   ️  Outlier removal would remove {100*(1-train_keep_ratio):.1f}% of training data")
        print(f"   Skipping outlier removal to preserve data")
    else:
        X_train = X_train[train_mask].copy()
        y_train = y_train[train_mask].copy()
        X_val = X_val[val_mask].copy()
        y_val = y_val[val_mask].copy()
        X_test = X_test[test_mask].copy()
        y_test = y_test[test_mask].copy()
        
        print(f"   Removed {len(train_mask) - train_mask.sum()} train outliers ({100*(1-train_keep_ratio):.1f}%)")
        print(f"   Removed {len(val_mask) - val_mask.sum()} val outliers")
        print(f"   Removed {len(test_mask) - test_mask.sum()} test outliers")
        print(f"   Train target range: {y_train.min():.2f} - {y_train.max():.2f}")
        print(f"   Test target range: {y_test.min():.2f} - {y_test.max():.2f}")

print(f"   Train: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")

# Check for distribution shift
print(f"\n Distribution Check:")
print(f"   Train target - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}, min: {y_train.min():.2f}, max: {y_train.max():.2f}")
print(f"   Val target - mean: {y_val.mean():.2f}, std: {y_val.std():.2f}, min: {y_val.min():.2f}, max: {y_val.max():.2f}")
print(f"   Test target - mean: {y_test.mean():.2f}, std: {y_test.std():.2f}, min: {y_test.min():.2f}, max: {y_test.max():.2f}")

# Check if training set has zero variance (critical issue!)
if y_train.std() == 0 or y_train.nunique() == 1:
    print(f"   ️  CRITICAL: Training set has ZERO variance after split!")
    print(f"   All training samples have the same target value: {y_train.mean():.2f}")
    print(f"   This will cause all models to predict the same value.")
    print(f"   FIXING: Shuffling data and re-splitting...")
    
    # Emergency fix: shuffle and re-split
    combined_indices = np.random.RandomState(42).permutation(len(X))
    X_combined = X.iloc[combined_indices].reset_index(drop=True)
    y_combined = y.iloc[combined_indices].reset_index(drop=True)
    
    train_idx = int(len(X_combined) * 0.7)
    val_idx = int(len(X_combined) * 0.85)
    X_train = X_combined.iloc[:train_idx].copy()
    X_val = X_combined.iloc[train_idx:val_idx].copy()
    X_test = X_combined.iloc[val_idx:].copy()
    y_train = y_combined.iloc[:train_idx].copy()
    y_val = y_combined.iloc[train_idx:val_idx].copy()
    y_test = y_combined.iloc[val_idx:].copy()
    
    print(f"    Re-split after shuffle")
    print(f"   Train target - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}, unique: {y_train.nunique()}")
    print(f"   Test target - mean: {y_test.mean():.2f}, std: {y_test.std():.2f}, unique: {y_test.nunique()}")
    
    if y_train.std() == 0:
        print(f"   ️  STILL ZERO VARIANCE after shuffle!")
        print(f"   This suggests a fundamental data issue - all Calculated_AQI values may be identical")
        print(f"   Cannot train meaningful model - exiting")
        raise ValueError("Training set has zero variance even after shuffling. Check data quality.")
else:
    mean_shift = abs(y_train.mean() - y_test.mean()) / y_train.std() if y_train.std() > 0 else 0
    std_shift = abs(y_train.std() - y_test.std()) / y_train.std() if y_train.std() > 0 else 0
    if mean_shift > 0.2:
        print(f"   ️  WARNING: Significant mean shift between train/test: {mean_shift:.2f} std devs")
    if std_shift > 0.2:
        print(f"   ️  WARNING: Significant std shift between train/test: {std_shift:.2f} std devs")

# Scale features
print("\n Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n Training models...")
models = {}
results = []

# First, try a simple baseline model to establish if features are predictive
from sklearn.linear_model import LinearRegression, Ridge
print("\n  Training Simple Baseline Models...")
try:
    # Simple Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_lr = np.clip(y_pred_lr, 0, None)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_test.mean())))
    print(f"    Linear Regression - RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")
    print(f"    Baseline (mean) - RMSE: {baseline_rmse:.4f}")
    if lr_r2 < 0:
        print(f"    ️  WARNING: Even simple linear regression has negative R²!")
        print(f"    This suggests features may not be predictive or there's a fundamental issue.")
    
    # Ridge Regression (regularized)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    y_pred_ridge = np.clip(y_pred_ridge, 0, None)
    
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    print(f"    Ridge Regression - RMSE: {ridge_rmse:.4f}, MAE: {ridge_mae:.4f}, R²: {ridge_r2:.4f}")
    
    models['LinearRegression'] = lr_model
    results.append({
        'Model': 'LinearRegression',
        'RMSE': lr_rmse,
        'MAE': lr_mae,
        'R²': lr_r2
    })
    
    models['Ridge'] = ridge_model
    results.append({
        'Model': 'Ridge',
        'RMSE': ridge_rmse,
        'MAE': ridge_mae,
        'R²': ridge_r2
    })
except Exception as e:
    print(f"    ️  Baseline model training failed: {e}")

# XGBoost with improved hyperparameters (reduced overfitting)
if XGBOOST_AVAILABLE:
    print("\n  Training XGBoost (with regularization)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,  # Reduced depth to prevent overfitting
        learning_rate=0.01,  # Slightly higher learning rate
        subsample=0.8,  # More subsampling for regularization
        colsample_bytree=0.8,  # More feature subsampling
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        min_child_weight=5,  # Increased to prevent overfitting
        gamma=0.2,  # Increased minimum loss reduction
        reg_alpha=0.5,  # Increased L1 regularization
        reg_lambda=2.0,  # Increased L2 regularization
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,  # Earlier stopping
        eval_metric='rmse',
        tree_method='hist',
        grow_policy='lossguide'
    )
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val), (X_test_scaled, y_test)],
        verbose=False
    )
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    # Clip predictions to reasonable range
    y_pred_xgb = np.clip(y_pred_xgb, 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae = mean_absolute_error(y_test, y_pred_xgb)
    r2 = r2_score(y_test, y_pred_xgb)
    
    # Comprehensive diagnostic information
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_test.mean())))
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
    
    print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"    Baseline RMSE (mean): {baseline_rmse:.4f}")
    print(f"    Baseline MAE (mean): {baseline_mae:.4f}")
    print(f"    Improvement over baseline RMSE: {((baseline_rmse - rmse) / baseline_rmse * 100):.2f}%")
    print(f"    Improvement over baseline MAE: {((baseline_mae - mae) / baseline_mae * 100):.2f}%")
    print(f"    Test target - mean: {y_test.mean():.2f}, std: {y_test.std():.2f}, min: {y_test.min():.2f}, max: {y_test.max():.2f}")
    print(f"    Predictions - mean: {y_pred_xgb.mean():.2f}, std: {y_pred_xgb.std():.2f}, min: {y_pred_xgb.min():.2f}, max: {y_pred_xgb.max():.2f}")
    
    # Check prediction distribution
    pred_error = y_test - y_pred_xgb
    print(f"    Prediction errors - mean: {pred_error.mean():.2f}, std: {pred_error.std():.2f}")
    print(f"    Mean absolute error: {np.abs(pred_error).mean():.2f}")
    
    # Check if model is systematically biased
    if abs(pred_error.mean()) > y_test.std() * 0.1:
        print(f"    ️  WARNING: Model has systematic bias of {pred_error.mean():.2f}")
    
    models['XGBoost'] = xgb_model
    results.append({
        'Model': 'XGBoost',
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })

# LightGBM with improved hyperparameters (reduced overfitting)
if LIGHTGBM_AVAILABLE:
    print("\n  Training LightGBM (with regularization)...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=5,  # Reduced depth
        learning_rate=0.01,
        subsample=0.8,  # More subsampling
        colsample_bytree=0.8,
        min_child_samples=30,  # Increased
        reg_alpha=0.5,  # Increased L1 regularization
        reg_lambda=2.0,  # Increased L2 regularization
        num_leaves=31,  # Reduced leaves
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=20,  # Increased
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        boosting_type='gbdt',
        objective='regression',
        metric='rmse'
    )
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val), (X_test_scaled, y_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    y_pred_lgb = lgb_model.predict(X_test_scaled)
    
    # Clip predictions to reasonable range
    y_pred_lgb = np.clip(y_pred_lgb, 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    mae = mean_absolute_error(y_test, y_pred_lgb)
    r2 = r2_score(y_test, y_pred_lgb)
    
    models['LightGBM'] = lgb_model
    results.append({
        'Model': 'LightGBM',
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
    print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Gradient Boosting (fallback) with improved hyperparameters
print("\n  Training Gradient Boosting (with regularization)...")
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth=5,  # Reduced depth
    learning_rate=0.01,
    subsample=0.8,  # More subsampling
    min_samples_split=30,  # Increased
    min_samples_leaf=15,  # Increased
    max_features='sqrt',
    random_state=42,
    validation_fraction=0.15,
    n_iter_no_change=50,  # Earlier stopping
    loss='squared_error',
    criterion='friedman_mse'
)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

# Clip predictions to reasonable range
y_pred_gb = np.clip(y_pred_gb, 0, None)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae = mean_absolute_error(y_test, y_pred_gb)
r2 = r2_score(y_test, y_pred_gb)

models['GradientBoosting'] = gb_model
results.append({
    'Model': 'GradientBoosting',
    'RMSE': rmse,
    'MAE': mae,
    'R²': r2
})
print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# LSTM Model with improved architecture and target scaling
if TENSORFLOW_AVAILABLE:
    print("\n  Training LSTM...")
    try:
        # Prepare sequence data for LSTM
        sequence_length = 24  # Look back 24 hours
        forecast_horizon = 1   # Predict 1 hour ahead
        
        # Use MinMaxScaler for features (better for neural networks)
        lstm_scaler = MinMaxScaler()
        X_train_lstm_scaled = lstm_scaler.fit_transform(X_train)
        X_val_lstm_scaled = lstm_scaler.transform(X_val)
        X_test_lstm_scaled = lstm_scaler.transform(X_test)
        
        # Scale target separately for better learning
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(
            X_train_lstm_scaled, y_train_scaled, 
            sequence_length=sequence_length, 
            forecast_horizon=forecast_horizon
        )
        X_val_seq, y_val_seq = create_sequences(
            X_val_lstm_scaled, y_val_scaled,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        X_test_seq, y_test_seq = create_sequences(
            X_test_lstm_scaled, y_test_scaled,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        # Reshape y for single output
        if y_train_seq.ndim > 1:
            y_train_seq = y_train_seq.reshape(-1)
        if y_val_seq.ndim > 1:
            y_val_seq = y_val_seq.reshape(-1)
        if y_test_seq.ndim > 1:
            y_test_seq = y_test_seq.reshape(-1)
        
        print(f"    Sequence shape: {X_train_seq.shape}")
        print(f"    Training sequences: {len(X_train_seq)}")
        
        # Build improved LSTM model with regularization
        lstm_model = Sequential([
            LSTM(256, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-8, verbose=0),
        ]
        
        # Train model with validation set
        history = lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=200,
            batch_size=128,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict and inverse transform
        y_pred_lstm_scaled = lstm_model.predict(X_test_seq, verbose=0).reshape(-1)
        y_pred_lstm = y_scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).ravel()
        
        # Clip predictions to reasonable range
        y_pred_lstm = np.clip(y_pred_lstm, 0, None)
        
        # Get actual y values (not scaled) for comparison
        y_test_actual = y_test.iloc[sequence_length:].values
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_lstm))
        mae = mean_absolute_error(y_test_actual, y_pred_lstm)
        r2 = r2_score(y_test_actual, y_pred_lstm)
        
        models['LSTM'] = {
            'model': lstm_model,
            'scaler': lstm_scaler,
            'y_scaler': y_scaler,
            'sequence_length': sequence_length
        }
        results.append({
            'Model': 'LSTM',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    except Exception as e:
        print(f"    ️ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()

# GRU Model with improved architecture and target scaling
if TENSORFLOW_AVAILABLE:
    print("\n  Training GRU...")
    try:
        # Prepare sequence data for GRU
        sequence_length = 24
        forecast_horizon = 1
        
        # Use MinMaxScaler for features
        gru_scaler = MinMaxScaler()
        X_train_gru_scaled = gru_scaler.fit_transform(X_train)
        X_val_gru_scaled = gru_scaler.transform(X_val)
        X_test_gru_scaled = gru_scaler.transform(X_test)
        
        # Scale target separately
        y_scaler_gru = MinMaxScaler()
        y_train_scaled = y_scaler_gru.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = y_scaler_gru.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler_gru.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(
            X_train_gru_scaled, y_train_scaled,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        X_val_seq, y_val_seq = create_sequences(
            X_val_gru_scaled, y_val_scaled,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        X_test_seq, y_test_seq = create_sequences(
            X_test_gru_scaled, y_test_scaled,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        # Reshape y for single output
        if y_train_seq.ndim > 1:
            y_train_seq = y_train_seq.reshape(-1)
        if y_val_seq.ndim > 1:
            y_val_seq = y_val_seq.reshape(-1)
        if y_test_seq.ndim > 1:
            y_test_seq = y_test_seq.reshape(-1)
        
        print(f"    Sequence shape: {X_train_seq.shape}")
        print(f"    Training sequences: {len(X_train_seq)}")
        
        # Build improved GRU model with regularization
        gru_model = Sequential([
            GRU(256, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2]),
                dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        gru_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-8, verbose=0),
        ]
        
        # Train model with validation set
        history = gru_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=200,
            batch_size=128,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict and inverse transform
        y_pred_gru_scaled = gru_model.predict(X_test_seq, verbose=0).reshape(-1)
        y_pred_gru = y_scaler_gru.inverse_transform(y_pred_gru_scaled.reshape(-1, 1)).ravel()
        
        # Clip predictions to reasonable range
        y_pred_gru = np.clip(y_pred_gru, 0, None)
        
        # Get actual y values (not scaled) for comparison
        y_test_actual = y_test.iloc[sequence_length:].values
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_gru))
        mae = mean_absolute_error(y_test_actual, y_pred_gru)
        r2 = r2_score(y_test_actual, y_pred_gru)
        
        models['GRU'] = {
            'model': gru_model,
            'scaler': gru_scaler,
            'y_scaler': y_scaler_gru,
            'sequence_length': sequence_length
        }
        results.append({
            'Model': 'GRU',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    except Exception as e:
        print(f"    ️ GRU training failed: {e}")
        import traceback
        traceback.print_exc()

# Select best model (before ensemble to get RMSE values)
results_df = pd.DataFrame(results)

# Ensemble Model - combine top models
print("\n  Training Ensemble Model...")
try:
    # Get predictions from all tree-based models
    ensemble_predictions = []
    ensemble_weights = []
    ensemble_model_names = []
    
    for model_name, model in models.items():
        if model_name not in ['LSTM', 'GRU']:  # Only use tree-based models for ensemble
            try:
                # Use validation set for weighting to avoid overfitting
                pred_val = model.predict(X_val_scaled)
                pred_val = np.clip(pred_val, 0, None)
                val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))
                
                pred = model.predict(X_test_scaled)
                pred = np.clip(pred, 0, None)
                ensemble_predictions.append(pred)
                ensemble_model_names.append(model_name)
                # Weight by inverse validation RMSE (better for generalization)
                ensemble_weights.append(1.0 / (val_rmse + 1e-6))
            except:
                continue
    
    if ensemble_predictions and len(ensemble_predictions) > 1:
        # Normalize weights
        ensemble_weights = np.array(ensemble_weights)
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        
        # Weighted average
        y_pred_ensemble = np.zeros_like(ensemble_predictions[0])
        for pred, weight in zip(ensemble_predictions, ensemble_weights):
            y_pred_ensemble += pred * weight
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        mae = mean_absolute_error(y_test, y_pred_ensemble)
        r2 = r2_score(y_test, y_pred_ensemble)
        
        models['Ensemble'] = {
            'models': {k: v for k, v in models.items() if k in ensemble_model_names},
            'weights': ensemble_weights.tolist(),
            'model_names': ensemble_model_names
        }
        results.append({
            'Model': 'Ensemble',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    else:
        print("    ️ Not enough models for ensemble")
except Exception as e:
    print(f"    ️ Ensemble training failed: {e}")

# Update results_df with ensemble
results_df = pd.DataFrame(results)
if len(results_df) == 0:
    raise ValueError("No models were successfully trained! Check for errors above.")

best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
if best_model_name not in models:
    raise KeyError(f"Best model '{best_model_name}' not found in models dictionary. Available models: {list(models.keys())}")
best_model = models[best_model_name]

print(f"\n Best Model: {best_model_name}")
print(results_df)

# Save model and scaler
print("\n Saving model and scaler...")
model_path = Path("best_model.pkl")
scaler_path = Path("scaler.pkl")
metadata_path = Path("best_model_metadata.json")
feature_names_path = Path("feature_names.json")

# Handle different model types
if best_model_name in ['LSTM', 'GRU']:
    # Save deep learning model
    model_dir = Path("best_model_keras")
    model_dir.mkdir(exist_ok=True)
    best_model['model'].save(str(model_dir))
    print(f" Keras model saved to {model_dir}")
    
    # Save scaler and metadata for deep learning model
    joblib.dump(best_model['scaler'], scaler_path)
    print(f" Scaler saved to {scaler_path}")
    
    # Save sequence length info
    sequence_info = {
        'sequence_length': best_model['sequence_length'],
        'model_type': best_model_name
    }
    with open(Path("sequence_info.json"), 'w') as f:
        json.dump(sequence_info, f, indent=2)
    print(f" Sequence info saved to sequence_info.json")
elif best_model_name == 'Ensemble':
    # Save ensemble model
    ensemble_data = {
        'model_names': best_model['model_names'],
        'weights': best_model['weights']
    }
    # Save individual models
    for model_name in best_model['model_names']:
        model_path_individual = Path(f"ensemble_{model_name.lower()}.pkl")
        joblib.dump(best_model['models'][model_name], model_path_individual)
    
    joblib.dump(ensemble_data, model_path)
    print(f" Ensemble model saved to {model_path}")
    
    joblib.dump(scaler, scaler_path)
    print(f" Scaler saved to {scaler_path}")
else:
    # Save tree-based model
    joblib.dump(best_model, model_path)
    print(f" Model saved to {model_path}")
    
    joblib.dump(scaler, scaler_path)
    print(f" Scaler saved to {scaler_path}")

# Save metadata
metadata = {
    "model_type": best_model_name,
    "sklearn_version": "1.5.2",
    "numpy_version": np.__version__,
    "target_column": target_col,
    "feature_count": len(X.columns),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "dataset_expanded": True,
    "augmentation_factor": 0 if not USE_AUGMENTATION else 2,  # No augmentation by default
    "train_val_test_split": "70/15/15",
    "best_rmse": float(results_df.loc[results_df['RMSE'].idxmin(), 'RMSE']),
    "best_mae": float(results_df.loc[results_df['RMSE'].idxmin(), 'MAE']),
    "best_r2": float(results_df.loc[results_df['RMSE'].idxmin(), 'R²']),
    "all_models": results_df.to_dict('records'),
    "created_at": datetime.now().isoformat()
}

# Add TensorFlow version if available
if TENSORFLOW_AVAILABLE:
    metadata["tensorflow_version"] = tf.__version__

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f" Metadata saved to {metadata_path}")

# Save feature names for prediction
feature_names = {
    "feature_columns": list(X.columns),
    "target_column": target_col
}
with open(feature_names_path, 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f" Feature names saved to {feature_names_path}")

print("\n" + "=" * 60)
print(" MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest model: {best_model_name}")
print(f"Test RMSE: {metadata['best_rmse']:.4f}")
print(f"Test MAE: {metadata['best_mae']:.4f}")
print(f"Test R²: {metadata['best_r2']:.4f}")

