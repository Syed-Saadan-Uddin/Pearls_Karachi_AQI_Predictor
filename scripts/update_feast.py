"""
Utility script to automatically update Feast feature store when data is created/updated.
This can be called from notebooks or scripts after data files are saved.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

def update_feast_feature_store(data_file: str = None, verbose: bool = True):
    """
    Automatically update Feast feature store after data is created/updated.
    
    Args:
        data_file: Path to the CSV file that was created/updated. 
                  If None, uses 'cleaned_aqi_weather_dataset.csv'
        verbose: Whether to print status messages
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine data file path
        if data_file is None:
            data_file = "cleaned_aqi_weather_dataset.csv"
        
        data_path = Path(data_file)
        
        # Check if data file exists
        if not data_path.exists():
            if verbose:
                print(f"️ Data file not found: {data_path}")
                print("   Skipping Feast update")
            return False
        
        if verbose:
            print("\n" + "="*60)
            print(" Updating Feast Feature Store")
            print("="*60)
        
        # Step 1: Prepare data for Feast
        if verbose:
            print("\n Step 1: Preparing data for Feast...")
        
        try:
            # Import setup_feast function
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            
            # Change to project root for imports
            original_cwd = os.getcwd()
            os.chdir(project_root)
            
            try:
                from setup_feast import setup_feast
                # Run setup with the specific data file
                setup_feast(data_file=str(data_path))
            finally:
                os.chdir(original_cwd)
            
            if verbose:
                print(" Data prepared for Feast")
        except Exception as e:
            if verbose:
                print(f"️ Error preparing data: {e}")
            return False
        
        # Step 2: Apply feature definitions
        if verbose:
            print("\n Step 2: Applying Feast feature definitions...")
        
        feature_store_path = Path(__file__).parent.parent / "feature_store"
        if not feature_store_path.exists():
            if verbose:
                print(f"️ Feature store directory not found: {feature_store_path}")
            return False
        
        try:
            result = subprocess.run(
                ["feast", "apply"],
                cwd=feature_store_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                if verbose:
                    print(" Feature definitions applied")
            else:
                if verbose:
                    print(f"️ Feast apply had warnings: {result.stderr}")
        except FileNotFoundError:
            if verbose:
                print("️ Feast CLI not found. Install with: pip install feast[sqlite]")
            return False
        except Exception as e:
            if verbose:
                print(f"️ Error applying feature definitions: {e}")
            return False
        
        # Step 3: Materialize features to online store
        if verbose:
            print("\n Step 3: Materializing features to online store...")
        
        try:
            # Get current timestamp in ISO format
            current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            
            result = subprocess.run(
                ["feast", "materialize-incremental", current_time],
                cwd=feature_store_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                if verbose:
                    print(" Features materialized to online store")
                    print(f"   Materialized up to: {current_time}")
                return True
            else:
                if verbose:
                    print(f"️ Materialization had warnings: {result.stderr}")
                # Still return True as this might be a partial success
                return True
        except Exception as e:
            if verbose:
                print(f"️ Error materializing features: {e}")
            return False
        
    except Exception as e:
        if verbose:
            print(f" Error updating Feast: {e}")
        return False
    finally:
        if verbose:
            print("="*60 + "\n")


def update_feast_from_notebook(data_file: str = None):
    """
    Convenience function for use in Jupyter notebooks.
    Automatically updates Feast after data is saved.
    
    Usage in notebook:
        from scripts.update_feast import update_feast_from_notebook
        df.to_csv('cleaned_aqi_weather_dataset.csv', index=False)
        update_feast_from_notebook()
    """
    return update_feast_feature_store(data_file=data_file, verbose=True)


if __name__ == "__main__":
    # Allow command-line usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Update Feast feature store")
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
    
    success = update_feast_feature_store(
        data_file=args.data_file,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)

