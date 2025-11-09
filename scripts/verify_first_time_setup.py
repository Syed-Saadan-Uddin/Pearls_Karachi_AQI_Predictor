"""
Script to verify first-time setup requirements
"""
import sys
from pathlib import Path

def check_file_exists(filepath, name, required=True):
    """Check if a file exists"""
    path = Path(filepath)
    exists = path.exists()
    status = "" if exists else ("" if required else "️")
    print(f"{status} {name}: {filepath}")
    if not exists and required:
        print(f"   ️  Required file missing!")
    return exists

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("First-Time Setup Verification")
    print("=" * 60)
    print()
    
    # Check required files
    print(" Required Files:")
    print("-" * 60)
    data_file = check_file_exists("cleaned_aqi_weather_dataset.csv", "Data file", required=True)
    model_file = check_file_exists("best_model.pkl", "Model file", required=False)
    scaler_file = check_file_exists("scaler.pkl", "Scaler file", required=False)
    metadata_file = check_file_exists("best_model_metadata.json", "Model metadata", required=False)
    print()
    
    # Check Python packages
    print(" Python Packages:")
    print("-" * 60)
    packages = {
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "sklearn": "Machine learning",
        "joblib": "Model serialization",
        "fastapi": "Backend API",
        "uvicorn": "ASGI server",
    }
    
    all_packages_ok = True
    for package, description in packages.items():
        installed = check_python_package(package)
        status = "" if installed else ""
        print(f"{status} {package}: {description}")
        if not installed:
            all_packages_ok = False
    print()
    
    # Summary and recommendations
    print("=" * 60)
    print("Summary & Recommendations:")
    print("=" * 60)
    
    if not data_file:
        print(" CRITICAL: Data file missing!")
        print("   → Run data extraction notebooks or obtain cleaned_aqi_weather_dataset.csv")
        print("   → See FIRST_TIME_SETUP.md for details")
        sys.exit(1)
    
    if not model_file:
        print("️  Model not trained yet!")
        print("   → Run: papermill model_train.ipynb /tmp/output.ipynb")
        print("   → Or open model_train.ipynb in Jupyter and run all cells")
    
    if not scaler_file:
        print("️  Scaler not created yet!")
        print("   → Run: python scripts/create_scaler.py")
    
    if not all_packages_ok:
        print(" Missing Python packages!")
        print("   → Run: pip install -r requirements-backend.txt")
    
    if data_file and model_file and scaler_file and all_packages_ok:
        print(" All requirements met! You're ready to start the application.")
        print()
        print("Next steps:")
        print("   1. Start backend:  cd backend && python main.py")
        print("   2. Start frontend: cd frontend && npm run dev")
        print("   3. Open browser:   http://localhost:3000")
    else:
        print()
        print(" For complete setup instructions, see FIRST_TIME_SETUP.md")

if __name__ == "__main__":
    main()

