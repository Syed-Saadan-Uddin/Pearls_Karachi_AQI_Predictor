"""
Hopsworks Feature Store utilities for the AQI Prediction API
"""
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables only

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("WARNING: Hopsworks not installed. Install with: pip install hopsworks")

# Global feature store instance
_feature_store: Optional[Any] = None
_feature_view: Optional[Any] = None


def get_feature_store() -> Optional[Any]:
    """
    Get or initialize the Hopsworks feature store instance.
    Raises RuntimeError if Hopsworks is not available or feature store cannot be initialized.
    """
    global _feature_store
    
    if _feature_store is not None:
        return _feature_store
    
    if not HOPSWORKS_AVAILABLE:
        raise RuntimeError(
            "Hopsworks is not available. "
            "Please install it with: pip install hopsworks"
        )
    
    try:
        # Initialize Hopsworks connection
        # Try to get API key from environment or use default project
        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_prediction")
        
        if api_key:
            project = hopsworks.login(api_key_value=api_key, project=project_name)
        else:
            # Try to use default connection (may require .hopsworksrc file)
            try:
                project = hopsworks.login(project=project_name)
            except Exception as login_error:
                raise RuntimeError(
                    f"No API key found and default login failed: {login_error}\n"
                    "Set HOPSWORKS_API_KEY environment variable or configure .hopsworksrc"
                ) from login_error
        
        fs = project.get_feature_store()
        _feature_store = fs
        print("Hopsworks feature store initialized successfully")
        return fs
            
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"Could not initialize Hopsworks feature store: {e}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(
            f"{error_msg}\n"
            "Please ensure:\n"
            "1. Hopsworks is installed: pip install hopsworks\n"
            "2. Set HOPSWORKS_API_KEY environment variable\n"
            "3. Set HOPSWORKS_PROJECT_NAME environment variable (default: aqi_prediction)\n"
            "4. Feature store is set up: python setup_hopsworks.py"
        ) from e


def get_feature_view() -> Optional[Any]:
    """
    Get or initialize the Hopsworks feature view.
    Raises RuntimeError if feature view cannot be initialized.
    """
    global _feature_view
    
    if _feature_view is not None:
        return _feature_view
    
    # get_feature_store() will raise RuntimeError if it cannot be initialized
    fs = get_feature_store()
    
    try:
        feature_view_name = os.getenv("HOPSWORKS_FEATURE_VIEW_NAME", "aqi_features")
        print(f"Attempting to get feature view '{feature_view_name}' (version 1)...")
        fv = fs.get_feature_view(name=feature_view_name, version=1)
        print(f"Feature view retrieved: {fv is not None}")
        
        # Check if feature view was actually retrieved
        if fv is None:
            # Try to list available feature views for debugging
            try:
                print("Attempting to list feature views for debugging...")
                # Some Hopsworks versions have get_feature_views() method
                if hasattr(fs, 'get_feature_views'):
                    views = fs.get_feature_views()
                    print(f"Available feature views: {[v.name for v in views] if views else 'None'}")
            except Exception as list_error:
                print(f"Could not list feature views: {list_error}")
            
            raise RuntimeError(
                f"\n Feature view '{feature_view_name}' (version 1) was not found in Hopsworks.\n\n"
                "To fix this, run the setup script:\n"
                f"   python setup_hopsworks.py\n\n"
                "Or if you want to insert data immediately:\n"
                f"   python setup_hopsworks.py --insert\n\n"
                "This will create the feature view and optionally insert data.\n"
                "If the feature view name is different, set HOPSWORKS_FEATURE_VIEW_NAME environment variable."
            )
        
        _feature_view = fv
        print(f"Hopsworks feature view '{feature_view_name}' (version 1) initialized successfully")
        return fv
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Could not get feature view '{feature_view_name}' (version 1): {error_msg}")
        raise RuntimeError(
            f"\n Could not get feature view '{feature_view_name}' (version 1): {error_msg}\n\n"
            "To fix this, run the setup script:\n"
            f"   python setup_hopsworks.py\n\n"
            "Or if you want to insert data immediately:\n"
            f"   python setup_hopsworks.py --insert\n\n"
            "This will create the feature view and optionally insert data.\n"
            "If the feature view name is different, set HOPSWORKS_FEATURE_VIEW_NAME environment variable."
        ) from e


def get_online_features(
    timestamps: List[int],
    feature_names: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Retrieve online features from Hopsworks for given timestamps.
    
    Args:
        timestamps: List of Unix timestamps (INT64)
        feature_names: Optional list of specific feature names to retrieve.
                       If None, retrieves all features from feature view.
    
    Returns:
        DataFrame with features, or None if feature store is not available
    """
    fv = get_feature_view()
    if fv is None:
        return None
    
    try:
        # Convert timestamps to datetime
        timestamps_dt = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create entity DataFrame for point-in-time join
        entity_df = pd.DataFrame({
            "timestamp": timestamps_dt
        })
        
        # Get online features using point-in-time join
        feature_vector = fv.get_batch_data(
            start_time=min(timestamps_dt),
            end_time=max(timestamps_dt),
            dataframe=entity_df
        )
        
        return feature_vector
        
    except Exception as e:
        print(f"WARNING: Error retrieving online features: {e}")
        return None


def get_offline_features(
    start_date: datetime,
    end_date: datetime,
    entity_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Retrieve offline features from Hopsworks for a time range.
    REQUIRED - Hopsworks must be available.
    
    Args:
        start_date: Start datetime for feature retrieval
        end_date: End datetime for feature retrieval
        entity_df: Optional DataFrame with entity values. If None, creates one from timestamps.
    
    Returns:
        DataFrame with features
    
    Raises:
        RuntimeError: If Hopsworks is not available or data retrieval fails
    """
    # get_feature_view() will raise RuntimeError if it cannot be initialized
    fv = get_feature_view()
    
    # Defensive check - should not be None if get_feature_view() succeeded
    if fv is None:
        raise RuntimeError(
            "Feature view is None. This should not happen if initialization succeeded. "
            "Please check Hopsworks connection and feature view configuration."
        )
    
    try:
        # If entity_df is provided, use it for point-in-time joins
        if entity_df is not None:
            # Ensure event_timestamp column exists
            if 'event_timestamp' not in entity_df.columns:
                if 'timestamp' in entity_df.columns:
                    entity_df['event_timestamp'] = pd.to_datetime(
                        entity_df['timestamp'], unit='s', errors='coerce'
                    )
                else:
                    raise ValueError("entity_df must have 'timestamp' or 'event_timestamp' column")
            
            # Get historical features using point-in-time join
            feature_df = fv.get_batch_data(
                start_time=start_date,
                end_time=end_date,
                dataframe=entity_df
            )
        else:
            # Get batch data for the time range
            feature_df = fv.get_batch_data(
                start_time=start_date,
                end_time=end_date
            )
        
        if feature_df is None or feature_df.empty:
            raise RuntimeError(
                f"Failed to retrieve features from Hopsworks for range {start_date} to {end_date}. "
                "Please ensure features are available in the feature store."
            )
        
        return feature_df
        
    except Exception as e:
        raise RuntimeError(
            f"Error retrieving offline features from Hopsworks: {e}"
        ) from e


def get_recent_features_for_prediction(hours: int = 24) -> pd.DataFrame:
    """
    Get recent features from Hopsworks for use in predictions.
    REQUIRED - Hopsworks must be available.
    
    Args:
        hours: Number of recent hours to retrieve
    
    Returns:
        DataFrame with recent features
    
    Raises:
        RuntimeError: If Hopsworks is not available or data retrieval fails
    """
    # get_feature_view() will raise RuntimeError if it cannot be initialized
    fv = get_feature_view()
    
    # Defensive check - should not be None if get_feature_view() succeeded
    if fv is None:
        raise RuntimeError(
            "Feature view is None. This should not happen if initialization succeeded. "
            "Please check Hopsworks connection and feature view configuration."
        )
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        features = get_offline_features(start_date, end_date)
        
        if features is None or features.empty:
            raise RuntimeError(
                f"Failed to retrieve recent features from Hopsworks (last {hours} hours). "
                "Please ensure features are available in the feature store."
            )
        
        return features
        
    except Exception as e:
        raise RuntimeError(
            f"Error retrieving recent features from Hopsworks: {e}"
        ) from e


def load_from_csv_fallback(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fallback function to load data from CSV if Hopsworks is not available.
    If the requested date range has no data, returns the most recent available data.
    """
    csv_path = Path(__file__).parent.parent / "cleaned_aqi_weather_dataset.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Create datetime column
    if 'event_timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['event_timestamp'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hour']].apply(
                lambda x: f"{int(x['year'])}-{int(x['month'])}-{int(x['day'])} {int(x['hour'])}:00:00",
                axis=1
            )
        )
    
    # Filter by date range
    filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()
    
    # If no data in requested range, return the most recent available data
    if filtered_df.empty:
        # Sort by datetime descending and get the most recent data
        df_sorted = df.sort_values('datetime', ascending=False)
        if not df_sorted.empty:
            # Get data up to the requested number of days
            days_diff = (end_date - start_date).days
            if days_diff > 0:
                # Get the most recent data points (approximately matching the requested days)
                # Limit to reasonable number of rows (e.g., days * 24 hours)
                max_rows = min(days_diff * 24, len(df_sorted))
                filtered_df = df_sorted.head(max_rows).copy()
            else:
                # If days_diff is 0 or negative, just return recent data
                filtered_df = df_sorted.head(100).copy()
    
    return filtered_df

