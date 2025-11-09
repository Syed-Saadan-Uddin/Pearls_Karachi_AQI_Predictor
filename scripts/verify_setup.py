"""
Script to verify that the setup is correct
"""
import sys
from pathlib import Path

def check_file(file_path, description):
    """Check if a file exists"""
    path = Path(file_path)
    if path.exists():
        print(f"[OK] {description}: {file_path}")
        return True
    else:
        print(f"[FAIL] {description} NOT FOUND: {file_path}")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print(f"[OK] Python package '{package_name}' is installed")
        return True
    except ImportError:
        print(f"[FAIL] Python package '{package_name}' is NOT installed")
        return False

def main():
    print("=" * 50)
    print("AQI Dashboard Setup Verification")
    print("=" * 50)
    print()
    
    all_ok = True
    
    # Check required files
    print("Checking required files...")
    all_ok &= check_file("cleaned_aqi_weather_dataset.csv", "Historical data file")
    all_ok &= check_file("best_model.pkl", "ML model file")
    
    # Check optional files
    print("\nChecking optional files...")
    scaler_exists = check_file("scaler.pkl", "Scaler file")
    if not scaler_exists:
        print("   [INFO] Run 'python scripts/create_scaler.py' to create scaler")
    
    # Check backend files
    print("\nChecking backend files...")
    all_ok &= check_file("backend/main.py", "Backend API")
    
    # Check frontend files
    print("\nChecking frontend files...")
    all_ok &= check_file("frontend/package.json", "Frontend package.json")
    all_ok &= check_file("frontend/src/App.jsx", "Frontend app")
    
    # Check Python packages
    print("\nChecking Python packages...")
    packages = ["fastapi", "pandas", "numpy", "sklearn", "joblib"]
    for pkg in packages:
        all_ok &= check_python_package(pkg)
    
    # Check Hopsworks integration
    print("\nChecking Hopsworks integration...")
    hopsworks_installed = False
    try:
        import hopsworks
        hopsworks_installed = True
        print("[OK] Hopsworks package is installed")
    except ImportError:
        print("[WARN] Hopsworks package is NOT installed")
        print("   [INFO] Install with: pip install hopsworks")
    
    # Check Hopsworks feature store setup
    if hopsworks_installed:
        # Check environment variables
        import os
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if api_key:
            print("[OK] HOPSWORKS_API_KEY environment variable is set")
        else:
            print("[WARN] HOPSWORKS_API_KEY environment variable is NOT set")
            print("   [INFO] Set it with: export HOPSWORKS_API_KEY=your_api_key")
        
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_prediction")
        print(f"[INFO] Using project name: {project_name}")
        
        # Check backend Hopsworks integration
        backend_hopsworks_utils = Path("backend/hopsworks_utils.py")
        if backend_hopsworks_utils.exists():
            print("[OK] Backend Hopsworks utilities found")
        else:
            print("[WARN] Backend Hopsworks utilities not found")
        
        # Check setup script
        setup_script = Path("setup_hopsworks.py")
        if setup_script.exists():
            print("[OK] Hopsworks setup script found")
        else:
            print("[WARN] Hopsworks setup script not found")
    else:
        print("[INFO] Hopsworks integration checks skipped (Hopsworks not installed)")
    
    # Check Node.js
    print("\nChecking Node.js...")
    import subprocess
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Node.js is installed: {result.stdout.strip()}")
        else:
            print("[FAIL] Node.js is NOT installed or not in PATH")
            all_ok = False
    except FileNotFoundError:
        print("[FAIL] Node.js is NOT installed or not in PATH")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("[SUCCESS] Setup verification PASSED!")
        print("\nNext steps:")
        print("1. Start backend: cd backend && python main.py")
        print("2. Start frontend: cd frontend && npm run dev")
        print("3. Open http://localhost:3000 in your browser")
    else:
        print("[ERROR] Setup verification FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 50)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

