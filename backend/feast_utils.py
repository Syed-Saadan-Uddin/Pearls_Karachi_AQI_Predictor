"""
Feast Feature Store utilities for the AQI Prediction API
"""
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from feast import FeatureStore

# Feast 0.56.0 should have better pandas 2.3.2 compatibility

# Global feature store instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> Optional[FeatureStore]:
    """
    Get or initialize the Feast feature store instance.
    Returns None if Feast is not available or feature store cannot be initialized.
    """
    global _feature_store
    
    if _feature_store is not None:
        return _feature_store
    
    try:
        # Get the feature store directory path
        feature_store_path = Path(__file__).parent.parent / "feature_store"
        
        if not feature_store_path.exists():
            print("WARNING: Feature store directory not found")
            return None
        
        # Change to feature store directory for Feast initialization
        original_cwd = os.getcwd()
        try:
            os.chdir(feature_store_path)
            fs = FeatureStore(repo_path=".")
            _feature_store = fs
            print("Feast feature store initialized successfully")
            return fs
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"WARNING: Could not initialize Feast feature store: {e}")
        return None


def get_online_features(
    timestamps: List[int],
    feature_names: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Retrieve online features from Feast for given timestamps.
    
    Args:
        timestamps: List of Unix timestamps (INT64)
        feature_names: Optional list of specific feature names to retrieve.
                       If None, retrieves all features from aqi_features view.
    
    Returns:
        DataFrame with features, or None if feature store is not available
    """
    fs = get_feature_store()
    if fs is None:
        return None
    
    try:
        # Prepare entity rows
        entity_rows = [{"timestamp": ts} for ts in timestamps]
        
        # Default to all features if not specified
        if feature_names is None:
            feature_names = [
                "aqi_features:aqi_index",
                "aqi_features:co",
                "aqi_features:no",
                "aqi_features:no2",
                "aqi_features:o3",
                "aqi_features:so2",
                "aqi_features:pm2_5",
                "aqi_features:pm10",
                "aqi_features:nh3",
                "aqi_features:temperature_2m",
                "aqi_features:relative_humidity_2m",
                "aqi_features:precipitation",
                "aqi_features:wind_speed_10m",
                "aqi_features:wind_direction_10m",
                "aqi_features:surface_pressure",
                "aqi_features:dew_point_2m",
                "aqi_features:apparent_temperature",
                "aqi_features:shortwave_radiation",
                "aqi_features:et0_fao_evapotranspiration",
                "aqi_features:year",
                "aqi_features:month",
                "aqi_features:day",
                "aqi_features:hour",
                "aqi_features:Calculated_AQI",
            ]
        
        # Retrieve features
        feature_vector = fs.get_online_features(
            features=feature_names,
            entity_rows=entity_rows
        )
        
        # Convert to DataFrame
        df = feature_vector.to_df()
        
        # Clean column names (remove 'aqi_features:' prefix)
        df.columns = [col.replace('aqi_features:', '') if 'aqi_features:' in col else col for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"WARNING: Error retrieving online features: {e}")
        return None


def get_offline_features(
    start_date: datetime,
    end_date: datetime,
    entity_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Retrieve offline features from Feast for a time range.
    REQUIRED - Feast must be available.
    
    Args:
        start_date: Start datetime for feature retrieval
        end_date: End datetime for feature retrieval
        entity_df: Optional DataFrame with entity values. If None, creates one from timestamps.
    
    Returns:
        DataFrame with features
    
    Raises:
        RuntimeError: If Feast is not available or data retrieval fails
    """
    fs = get_feature_store()
    if fs is None:
        raise RuntimeError(
            "Feast feature store is not available. "
            "Please ensure Feast is properly initialized."
        )
    
    # Get the feature store directory path
    feature_store_path = Path(__file__).parent.parent / "feature_store"
    original_cwd = os.getcwd()
    
    try:
        # Change to feature store directory for Feast operations
        os.chdir(feature_store_path)
        
        # If no entity_df provided, create one with timestamps
        if entity_df is None:
            timestamps = []
            event_timestamps = []
            current = start_date
            while current <= end_date:
                timestamps.append(int(current.timestamp()))
                event_timestamps.append(current)
                current += timedelta(hours=1)
            
            # Create entity_df with both timestamp and event_timestamp
            # Match the format in parquet file: datetime64[ns] (not DatetimeArray)
            entity_df = pd.DataFrame({
                "timestamp": timestamps
            })
            # Workaround for Feast 0.47.0 + pandas 2.3.2 DatetimeArray incompatibility
            # Create timestamps as individual values, then use pd.Series constructor with explicit dtype
            # This avoids DatetimeArray creation
            timestamp_values = []
            for dt in event_timestamps:
                if isinstance(dt, datetime):
                    ts = pd.Timestamp(dt)
                else:
                    ts = pd.Timestamp(dt)
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                # Convert to numpy datetime64 scalar (not array)
                timestamp_values.append(ts.value)  # Use .value to get int64 nanoseconds
            
            # Create Series from int64 values, then convert to datetime64[ns]
            # This bypasses DatetimeArray creation
            entity_df['event_timestamp'] = pd.Series(
                pd.to_datetime(timestamp_values, unit='ns'),
                index=entity_df.index
            )
        else:
            # If entity_df is provided, ensure event_timestamp is in the correct format
            if 'event_timestamp' not in entity_df.columns:
                # Create event_timestamp from timestamp if not present
                entity_df['event_timestamp'] = pd.to_datetime(entity_df['timestamp'], unit='s')
            
            # Workaround for Feast 0.47.0 + pandas 2.3.2 DatetimeArray incompatibility
            # Convert to int64 nanoseconds, then back to datetime to avoid DatetimeArray
            timestamp_values = []
            for dt in entity_df['event_timestamp']:
                ts = pd.Timestamp(dt)
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                timestamp_values.append(ts.value)  # Use .value to get int64 nanoseconds
            
            # Create Series from int64 values, then convert to datetime64[ns]
            entity_df['event_timestamp'] = pd.Series(
                pd.to_datetime(timestamp_values, unit='ns'),
                index=entity_df.index
            )
        
        # Retrieve offline features using Feast only
        # Workaround for DatetimeArray comparison issue: try multiple formats
        feature_list = [
            "aqi_features:aqi_index",
            "aqi_features:co",
            "aqi_features:no",
            "aqi_features:no2",
            "aqi_features:o3",
            "aqi_features:so2",
            "aqi_features:pm2_5",
            "aqi_features:pm10",
            "aqi_features:nh3",
            "aqi_features:temperature_2m",
            "aqi_features:relative_humidity_2m",
            "aqi_features:precipitation",
            "aqi_features:wind_speed_10m",
            "aqi_features:wind_direction_10m",
            "aqi_features:surface_pressure",
            "aqi_features:dew_point_2m",
            "aqi_features:apparent_temperature",
            "aqi_features:shortwave_radiation",
            "aqi_features:et0_fao_evapotranspiration",
            "aqi_features:year",
            "aqi_features:month",
            "aqi_features:day",
            "aqi_features:hour",
            "aqi_features:Calculated_AQI",
        ]
        
        # Try with current format first
        try:
            feature_df = fs.get_historical_features(
                entity_df=entity_df,
                features=feature_list
            ).to_df()
        except (TypeError, ValueError) as e:
            error_str = str(e)
            if "DatetimeArray" in error_str or "datetime64" in error_str:
                # Try workaround: convert event_timestamp to string and back
                # This forces pandas to recreate the Series without DatetimeArray
                entity_df_copy = entity_df.copy()
                # Convert to string then back to datetime to break DatetimeArray chain
                entity_df_copy['event_timestamp'] = pd.to_datetime(
                    entity_df_copy['event_timestamp'].astype(str)
                )
                try:
                    feature_df = fs.get_historical_features(
                        entity_df=entity_df_copy,
                        features=feature_list
                    ).to_df()
                except Exception as e2:
                    # If that also fails, try with Python datetime objects
                    entity_df_copy2 = entity_df.copy()
                    python_dts = [pd.Timestamp(dt).to_pydatetime() for dt in entity_df_copy2['event_timestamp']]
                    entity_df_copy2['event_timestamp'] = pd.Series(python_dts, dtype='object', index=entity_df_copy2.index)
                    try:
                        feature_df = fs.get_historical_features(
                            entity_df=entity_df_copy2,
                            features=feature_list
                        ).to_df()
                    except Exception as e3:
                        # Last resort: provide detailed error
                        raise RuntimeError(
                            f"Feast datetime compatibility error after multiple attempts: {error_str}\n"
                            f"Attempt 1 dtype: {entity_df['event_timestamp'].dtype}\n"
                            f"Attempt 2 dtype: {entity_df_copy['event_timestamp'].dtype}\n"
                            f"Attempt 3 dtype: {entity_df_copy2['event_timestamp'].dtype}\n"
                            f"Original error: {e}\n"
                            f"String conversion error: {e2}\n"
                            f"Object dtype error: {e3}\n"
                            f"This appears to be a Feast/pandas version incompatibility. "
                            f"Consider updating Feast or using a compatible pandas version."
                        ) from e3
            else:
                raise
        
        # Clean column names
        feature_df.columns = [col.replace('aqi_features:', '') if 'aqi_features:' in col else col for col in feature_df.columns]
        
        if feature_df is None or feature_df.empty:
            raise RuntimeError(
                f"Failed to retrieve features from Feast for range {start_date} to {end_date}. "
                "Please ensure features are materialized."
            )
        
        return feature_df
        
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"Error retrieving offline features from Feast: {e}"
        ) from e
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)


def get_recent_features_for_prediction(hours: int = 24) -> pd.DataFrame:
    """
    Get recent features from Feast for use in predictions.
    REQUIRED - Feast must be available.
    
    Args:
        hours: Number of recent hours to retrieve
    
    Returns:
        DataFrame with recent features
    
    Raises:
        RuntimeError: If Feast is not available or data retrieval fails
    """
    fs = get_feature_store()
    
    if fs is None:
        raise RuntimeError(
            "Feast feature store is not available. "
            "Please ensure Feast is properly initialized."
        )
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        # get_offline_features already handles directory changes
        features = get_offline_features(start_date, end_date)
        
        if features is None or features.empty:
            raise RuntimeError(
                f"Failed to retrieve recent features from Feast (last {hours} hours). "
                "Please ensure features are materialized."
            )
        
        return features
        
    except Exception as e:
        raise RuntimeError(
            f"Error retrieving recent features from Feast: {e}"
        ) from e

