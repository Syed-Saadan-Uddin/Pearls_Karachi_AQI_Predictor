"""
Helper functions for use in Jupyter notebooks.
Automatically updates Hopsworks when data is saved.
"""
import sys
from pathlib import Path

def auto_update_hopsworks(data_file: str = "cleaned_aqi_weather_dataset.csv"):
    """
    Automatically update Hopsworks feature store after saving data.
    
    Usage in notebook:
        import sys
        sys.path.append('.')
        from notebook_helpers import auto_update_hopsworks
        
        # After saving data
        df.to_csv('cleaned_aqi_weather_dataset.csv', index=False)
        auto_update_hopsworks()
    
    Args:
        data_file: Path to the CSV file that was just saved
    """
    try:
        from scripts.update_hopsworks import update_hopsworks_from_notebook
        return update_hopsworks_from_notebook(data_file=data_file)
    except ImportError as e:
        print(f"️ Could not import Hopsworks update function: {e}")
        print("   Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"️ Error updating Hopsworks: {e}")
        return False

# Alias for backward compatibility
auto_update_feast = auto_update_hopsworks

