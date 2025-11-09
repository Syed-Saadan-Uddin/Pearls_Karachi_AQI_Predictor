"""
Setup script for Hopsworks feature store
"""
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print(" Hopsworks not installed. Install with: pip install hopsworks")


def setup_hopsworks(data_file: str = None, auto_insert: bool = False):
    """Prepare data and setup Hopsworks feature store"""
    
    # Load environment variables from .env file if available
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    if not HOPSWORKS_AVAILABLE:
        print(" Hopsworks is not installed. Please install it first:")
        print("   pip install hopsworks")
        return
    
    # Check for API key
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print(" HOPSWORKS_API_KEY environment variable not set")
        print("\n   You can set it in one of these ways:")
        print("   1. Create a .env file in the project root with:")
        print("      HOPSWORKS_API_KEY=your_api_key")
        print("      HOPSWORKS_PROJECT_NAME=aqi_prediction")
        print("\n   2. Set it as an environment variable:")
        print("      Windows (PowerShell): $env:HOPSWORKS_API_KEY='your_api_key'")
        print("      Windows (CMD): set HOPSWORKS_API_KEY=your_api_key")
        print("      Linux/Mac: export HOPSWORKS_API_KEY=your_api_key")
        return
    
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_prediction")
    
    # Prepare data for Hopsworks
    print("Preparing data for Hopsworks...")
    
    # Use provided data file or default
    if data_file is None:
        data_path = Path("cleaned_aqi_weather_dataset.csv")
    else:
        data_path = Path(data_file)
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return
    
    df = pd.read_csv(data_path)
    
    # Validate required columns exist
    required_cols = ['year', 'month', 'day', 'hour']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f" Error: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Create event_timestamp from year, month, day, hour columns
    df['event_timestamp'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']], 
        errors='coerce'
    )
    
    # Check for invalid timestamps
    invalid_timestamps = df['event_timestamp'].isna().sum()
    if invalid_timestamps > 0:
        print(f" Warning: {invalid_timestamps} rows have invalid timestamps")
        df = df.dropna(subset=['event_timestamp'])
    
    # Create created timestamp
    df['created'] = datetime.now()
    
    # Ensure timestamp column exists for entity (must be datetime for Hopsworks)
    df['timestamp'] = df['event_timestamp']
    
    # Remove any rows with null timestamps
    df = df.dropna(subset=['timestamp', 'event_timestamp'])
    
    if len(df) == 0:
        print(" Error: No valid data rows after processing")
        return
    
    print(f" Prepared data: {len(df)} rows")
    print(f"   Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
    print(f"   Columns: {len(df.columns)}")
    
    # Connect to Hopsworks
    print("\n Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=api_key, project=project_name)
        fs = project.get_feature_store()
        print(" Connected to Hopsworks")
    except Exception as e:
        print(f" Failed to connect to Hopsworks: {e}")
        return
    
    # Create feature group if it doesn't exist
    print("\n Setting up feature group...")
    try:
        feature_group_name = "aqi_weather_features"
        fg = None
        
        # Check if feature group exists
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if fg is not None:
                print(f" Feature group '{feature_group_name}' already exists")
            else:
                raise ValueError("Feature group returned None")
        except Exception as get_error:
            # Feature group doesn't exist, create new one
            print(f"   Feature group not found (error: {get_error}), creating new one...")
            try:
                # Create new feature group
                # IMPORTANT: online_enabled=False because timestamp/datetime is not supported as primary key for online stores
                # This is fine for our use case - we use offline storage for batch processing and historical queries
                # If you need online features, you would need to use a different primary key (e.g., integer ID)
                fg = fs.create_feature_group(
                    name=feature_group_name,
                    version=1,
                    description="AQI and weather features for prediction",
                    primary_key=["timestamp"],
                    event_time="event_timestamp",
                    online_enabled=False  # Must be False - timestamp not supported as PK for online stores
                )
                
                # Verify it was created
                if fg is None:
                    raise ValueError("Failed to create feature group - returned None")
                
                # Save the feature group (some Hopsworks versions require this)
                try:
                    fg.save()
                except AttributeError:
                    # save() might not exist in all versions, that's okay
                    pass
                except Exception as save_error:
                    print(f"   Warning: Could not save feature group: {save_error}")
                
                print(f" Created feature group '{feature_group_name}'")
            except Exception as create_error:
                print(f" Failed to create feature group: {create_error}")
                import traceback
                traceback.print_exc()
                raise
        
        # Verify feature group is valid before using it
        if fg is None:
            raise ValueError("Feature group is None - cannot proceed")
        
        # Insert data if requested
        if auto_insert:
            print("\n Inserting data into feature group...")
            print(f"   Preparing to insert {len(df)} rows...")
            try:
                # Ensure timestamp is datetime type
                if df['timestamp'].dtype != 'datetime64[ns]':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['event_timestamp'].dtype != 'datetime64[ns]':
                    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
                
                # Insert data (automatically commits in newer Hopsworks versions)
                fg.insert(df)
                print(" Data inserted successfully")
            except Exception as insert_error:
                print(f" Failed to insert data: {insert_error}")
                print(f"   Data shape: {df.shape}")
                print(f"   Data columns: {list(df.columns)}")
                print(f"   Timestamp dtype: {df['timestamp'].dtype if 'timestamp' in df.columns else 'MISSING'}")
                import traceback
                traceback.print_exc()
                raise
        
        # Create feature view if it doesn't exist
        print("\n Setting up feature view...")
        try:
            feature_view_name = "aqi_features"
            fv = None
            try:
                fv = fs.get_feature_view(name=feature_view_name, version=1)
                # Check if feature view was actually retrieved
                if fv is None:
                    print(f" Feature view '{feature_view_name}' returned None, creating new one...")
                    raise ValueError("Feature view returned None")
                print(f" Feature view '{feature_view_name}' already exists")
            except (ValueError, AttributeError, Exception) as get_error:
                # Feature view doesn't exist or returned None, create new one
                print(f"   Feature view not found or invalid (error: {get_error}), creating new one...")
                try:
                    fv = fs.create_feature_view(
                        name=feature_view_name,
                        version=1,
                        query=fg.select_all(),
                        description="AQI and weather features for prediction"
                    )
                    # Verify it was created
                    if fv is None:
                        raise ValueError("Failed to create feature view - returned None")
                    
                    # Try to save if the method exists (not all Hopsworks versions have save())
                    try:
                        if hasattr(fv, 'save'):
                            fv.save()
                    except AttributeError:
                        # save() doesn't exist, that's okay - feature view is already created
                        pass
                    except Exception as save_error:
                        print(f"   Warning: Could not save feature view: {save_error}")
                        # Continue anyway - feature view might still be created
                    
                    # Verify the feature view was actually created by trying to retrieve it
                    try:
                        verify_fv = fs.get_feature_view(name=feature_view_name, version=1)
                        if verify_fv is None:
                            raise ValueError("Feature view was created but cannot be retrieved")
                        print(f" Created feature view '{feature_view_name}' and verified it exists")
                    except Exception as verify_error:
                        print(f" Feature view created but verification failed: {verify_error}")
                        print(f"   The feature view might still be available. Please check manually.")
                        # Don't raise - feature view might still work
                except Exception as create_error:
                    print(f" Failed to create feature view: {create_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Final verification
            if fv is None:
                raise ValueError("Feature view is None after setup - cannot proceed")
                
        except Exception as e:
            print(f" Could not setup feature view: {e}")
            import traceback
            traceback.print_exc()
            print("   You can create it manually later")
        
        if not auto_insert:
            print("\n Hopsworks setup complete!")
            print("\nNext steps:")
            print("1. Insert data into feature group:")
            print(f"   python setup_hopsworks.py --insert")
            print("\n   Or manually:")
            print(f"   fg = fs.get_feature_group(name='{feature_group_name}', version=1)")
            print("   fg.insert(df)")
    
    except Exception as e:
        print(f" Error setting up feature group: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup Hopsworks feature store")
    parser.add_argument("--data-file", type=str, help="Path to data CSV file")
    parser.add_argument("--insert", action="store_true", help="Automatically insert data after setup")
    args = parser.parse_args()
    
    setup_hopsworks(data_file=args.data_file, auto_insert=args.insert)

