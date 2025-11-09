"""
Utility script to automatically update Hopsworks feature store when data is created/updated.
This can be called from notebooks or scripts after data files are saved.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

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


def update_hopsworks_feature_store(data_file: str = None, verbose: bool = True):
    """
    Automatically update Hopsworks feature store after data is created/updated.
    
    Args:
        data_file: Path to the CSV file that was created/updated. 
                  If None, uses 'cleaned_aqi_weather_dataset.csv'
        verbose: Whether to print status messages
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not HOPSWORKS_AVAILABLE:
        if verbose:
            print("️ Hopsworks not installed. Install with: pip install hopsworks")
        return False
    
    try:
        # Check for API key
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            if verbose:
                print("️ HOPSWORKS_API_KEY environment variable not set")
            return False
        
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_prediction")
        
        # Determine data file path
        if data_file is None:
            data_file = "cleaned_aqi_weather_dataset.csv"
        
        data_path = Path(data_file)
        
        # Check if data file exists
        if not data_path.exists():
            if verbose:
                print(f"️ Data file not found: {data_path}")
                print("   Skipping Hopsworks update")
            return False
        
        if verbose:
            print("\n" + "="*60)
            print(" Updating Hopsworks Feature Store")
            print("="*60)
        
        # Step 1: Prepare data
        if verbose:
            print("\n Step 1: Preparing data for Hopsworks...")
        
        import pandas as pd
        
        try:
            df = pd.read_csv(data_path)
            
            # Create event_timestamp
            df['event_timestamp'] = pd.to_datetime(
                df[['year', 'month', 'day', 'hour']], 
                errors='coerce'
            )
            df['created'] = datetime.now()
            df['timestamp'] = df['event_timestamp']
            
            if verbose:
                print(f" Data prepared: {len(df)} rows")
        except Exception as e:
            if verbose:
                print(f"️ Error preparing data: {e}")
            return False
        
        # Step 2: Connect to Hopsworks
        if verbose:
            print("\n Step 2: Connecting to Hopsworks...")
        
        try:
            project = hopsworks.login(api_key_value=api_key, project=project_name)
            fs = project.get_feature_store()
            if verbose:
                print(" Connected to Hopsworks")
        except Exception as e:
            if verbose:
                print(f"️ Error connecting to Hopsworks: {e}")
            return False
        
        # Step 3: Get or create feature group
        if verbose:
            print("\n Step 3: Getting feature group...")
        
        feature_group_name = "aqi_weather_features"
        
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=1)
            if verbose:
                print(f" Feature group '{feature_group_name}' found")
        except:
            if verbose:
                print(f"️ Feature group '{feature_group_name}' not found")
                print("   Please run setup_hopsworks.py first")
            return False
        
        # Step 4: Insert data
        if verbose:
            print("\n Step 4: Inserting data into feature group...")
        
        try:
            fg.insert(df)
            if verbose:
                print(" Data inserted successfully")
                print(f"   Inserted {len(df)} rows")
            return True
        except Exception as e:
            if verbose:
                print(f"️ Error inserting data: {e}")
            return False
        
    except Exception as e:
        if verbose:
            print(f" Error updating Hopsworks: {e}")
        return False
    finally:
        if verbose:
            print("="*60 + "\n")


def update_hopsworks_from_notebook(data_file: str = None):
    """
    Convenience function for use in Jupyter notebooks.
    Automatically updates Hopsworks after data is saved.
    
    Usage in notebook:
        from scripts.update_hopsworks import update_hopsworks_from_notebook
        df.to_csv('cleaned_aqi_weather_dataset.csv', index=False)
        update_hopsworks_from_notebook()
    """
    return update_hopsworks_feature_store(data_file=data_file, verbose=True)


if __name__ == "__main__":
    # Allow command-line usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Update Hopsworks feature store")
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to CSV file that was created/updated"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    success = update_hopsworks_feature_store(
        data_file=args.data_file,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)

