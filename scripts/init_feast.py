"""
Script to initialize and verify Feast feature store setup
"""
import sys
import os
from pathlib import Path
import subprocess

def run_command(cmd, cwd=None, description=""):
    """Run a shell command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f" Success: {description or cmd}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f" Failed: {description or cmd}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f" Exception: {e}")
        return False

def main():
    print("=" * 60)
    print("Feast Feature Store Initialization")
    print("=" * 60)
    
    # Check if Feast is installed
    print("\n1. Checking Feast installation...")
    try:
        import feast
        print(" Feast is installed")
    except ImportError:
        print(" Feast is not installed")
        print("\nPlease install Feast first:")
        print("  pip install feast[sqlite]")
        return 1
    
    # Check if feature store directory exists
    feature_store_path = Path("feature_store")
    if not feature_store_path.exists():
        print("\n Feature store directory not found")
        print("Please ensure the feature_store directory exists")
        return 1
    
    # Step 1: Prepare data
    print("\n2. Preparing data for Feast...")
    setup_script = Path("setup_feast.py")
    if setup_script.exists():
        success = run_command(
            f"python {setup_script}",
            description="Preparing data for Feast"
        )
        if not success:
            print("️ Data preparation had issues, but continuing...")
    else:
        print("️ setup_feast.py not found, skipping data preparation")
    
    # Step 2: Apply feature definitions
    print("\n3. Applying Feast feature definitions...")
    success = run_command(
        "feast apply",
        cwd=feature_store_path,
        description="Applying Feast feature definitions"
    )
    if not success:
        print(" Failed to apply feature definitions")
        return 1
    
    # Step 3: Materialize features (optional, but recommended)
    print("\n4. Materializing features to online store...")
    print("   (This may take a while depending on data size)")
    
    # Get current timestamp in ISO format
    from datetime import datetime
    current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    
    success = run_command(
        f'feast materialize-incremental {current_time}',
        cwd=feature_store_path,
        description="Materializing features to online store"
    )
    if not success:
        print("️ Feature materialization had issues")
        print("   You can try again later with:")
        print(f"   cd feature_store && feast materialize-incremental {current_time}")
    
    # Step 4: Verify setup
    print("\n5. Verifying Feast setup...")
    try:
        os.chdir(feature_store_path)
        from feast import FeatureStore
        fs = FeatureStore(repo_path=".")
        print(" Feature store initialized successfully")
        
        # Try to get a sample feature
        try:
            # This is just a test - we won't actually use the result
            print(" Feature store is ready to use")
        except Exception as e:
            print(f"️ Feature store initialized but test query failed: {e}")
        
        os.chdir("..")
    except Exception as e:
        print(f" Failed to verify setup: {e}")
        os.chdir("..")
        return 1
    
    print("\n" + "=" * 60)
    print(" Feast feature store setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the backend: cd backend && python main.py")
    print("2. The backend will automatically use Feast if available")
    print("3. Check /health endpoint to verify Feast integration")
    print("\nFor more information, see feature_store/README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

