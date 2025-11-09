# AQI Prediction and Visualization Platform

This project is a full-stack application designed to predict and visualize the Air Quality Index (AQI). It features a machine learning pipeline that ingests weather and AQI data, trains a predictive model, and serves the predictions through a web-based dashboard.

## Key Features

- **Real-time AQI Display**: View the most current AQI data for a specific location.
- **AQI Forecasting**: Get predictions for future AQI values.
- **Data-driven Insights**: The backend is powered by a LightGBM model trained on historical weather and AQI data.
- **Automated ML Pipelines**: GitHub Actions are configured to run daily and hourly to retrain the model and ingest new data, ensuring the predictions are always based on the latest information.
- **Modern Frontend**: A responsive and intuitive dashboard built with React and Tailwind CSS.

## Technologies Used

- **Frontend**: React, Vite, Tailwind CSS
- **Backend**: Python, FastAPI
- **Machine Learning**:
  - **Model**: LightGBM, XGBoost, LSTM, GRU
  - **Libraries**: Scikit-learn, Pandas, NumPy
  - **Notebooks**: Jupyter for Exploratory Data Analysis (EDA) and model experimentation.
- **MLOps & Data Pipeline**:
  - **Feature Store**: Hopsworks 
  - **CI/CD**: GitHub Actions for automated model training and data ingestion.

## Project Structure

```
.
├── .github/workflows/    # GitHub Actions for CI/CD pipelines
├── backend/              # FastAPI application for serving predictions
├── frontend/             # React-based frontend dashboard
├── scripts/              # Utility scripts for setup and execution
├── *.ipynb               # Jupyter notebooks for EDA and model training
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js and npm
- An account on [Hopsworks.ai](https://www.hopsworks.ai/) for the feature store.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-backend.txt
    ```

3.  **Install frontend dependencies:**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

4.  **Initial Setup:**
    Follow the instructions in `FIRST_TIME_SETUP.md` to configure your environment, set up the Hopsworks feature store, and prepare the initial dataset and model.

## Usage

1.  **Start the Backend Server:**
    - On Windows: `.\scripts\start_backend.bat`
    - On Linux/macOS: `sh ./scripts/start_backend.sh`
    The backend will be running at `http://127.0.0.1:8000`.

2.  **Start the Frontend Application:**
    ```bash
    cd frontend
    npm run dev
    ```
    The frontend will be accessible at `http://localhost:3000`.

## CI/CD Automation

This project uses GitHub Actions to automate key parts of the ML lifecycle:

-   **Hourly Data Pipeline (`hourly-data-pipeline.yml`):** This workflow runs every hour to fetch the latest data and update the feature store.
-   **Daily Model Pipeline (`daily-model-pipeline.yml`):** This workflow runs daily to retrain the predictive model with the latest data from the feature store, ensuring the model's accuracy over time.

# Complete Setup Guide - From Scratch

This guide will walk you through setting up and running the AQI Prediction Dashboard from absolute scratch, assuming you have **no data, models, scalers, or anything**.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Node.js 16+** and **npm** installed
- **Git** installed (if cloning the repository)
- **Hopsworks Account** (free tier available at [hopsworks.ai](https://hopsworks.ai))

Verify your installations:
```bash
python --version  # Should be 3.8 or higher
node --version    # Should be 16 or higher
npm --version     # Should be 6 or higher
```

---

## 🚀 Step-by-Step Setup

### Step 1: Clone and Navigate to Project

```bash
# If you haven't cloned yet
git clone <repository-url>

```

### Step 2: Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Python Dependencies

```bash
# Install backend dependencies (includes Hopsworks)
pip install -r requirements-backend.txt

# Install additional dependencies for notebooks and training
pip install jupyter papermill ipykernel
```

Verify Hopsworks installation:
```bash
python -c "import hopsworks; print('✅ Hopsworks installed')"
```

### Step 4: Set Up Hopsworks Account

1. **Create a Hopsworks account** (if you don't have one):
   - Go to [hopsworks.ai](https://hopsworks.ai)
   - Sign up for a free account
   - Create a new project (or use default)

2. **Get your API key**:
   - In Hopsworks UI, go to your profile → API Keys
   - Create a new API key
   - Copy the API key

3. **Set environment variables** (Choose one method):

   **Method 1: Using .env file (Recommended - Easiest)**
   
   Create a `.env` file in the project root:
   ```bash
   # Copy the example file
   cp .env.example .env
   ```
   
   Then edit `.env` and add your API key:
   ```
   HOPSWORKS_API_KEY=your_api_key_here
   HOPSWORKS_PROJECT_NAME=aqi_prediction
   ```
   
   **Method 2: Environment Variables (Temporary - Current Session Only)**
   
   **Windows (Command Prompt):**
   ```cmd
   set HOPSWORKS_API_KEY=your_api_key_here
   set HOPSWORKS_PROJECT_NAME=aqi_prediction
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:HOPSWORKS_API_KEY="your_api_key_here"
   $env:HOPSWORKS_PROJECT_NAME="aqi_prediction"
   ```

   **macOS/Linux:**
   ```bash
   export HOPSWORKS_API_KEY=your_api_key_here
   export HOPSWORKS_PROJECT_NAME=aqi_prediction
   ```

   **Method 3: Permanent Environment Variables**
   
   **Linux/Mac** (add to `~/.bashrc` or `~/.zshrc`):
   ```bash
   echo 'export HOPSWORKS_API_KEY=your_api_key_here' >> ~/.bashrc
   echo 'export HOPSWORKS_PROJECT_NAME=aqi_prediction' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   **Windows**: Use System Properties → Environment Variables

### Step 5: Prepare Your Data

You need data to train the model. You have two options:

#### Option A: Use Sample Data (If Available)

If you have a sample `cleaned_aqi_weather_dataset.csv` file, place it in the project root and skip to Step 6.

#### Option B: Generate Data from APIs (Recommended)

1. **Fetch historical data** (this may take 10-30 minutes):
   ```bash
   # Open Jupyter notebook
   jupyter notebook
   ```
   - Open `feature_extract_from_api_hourly.ipynb`
   - Run all cells (this fetches data from APIs)
   - This creates `historical_aqi_weather_data.json`

2. **Extract and process the data**:
   - Open `extract_data.ipynb` in Jupyter
   - Run all cells
   - This creates `historical_data.csv`

3. **Clean and prepare the data**:
   - Open `EDA.ipynb` in Jupyter
   - Run all cells
   - This creates `cleaned_aqi_weather_dataset.csv`

**Verify data file exists:**
```bash
ls cleaned_aqi_weather_dataset.csv  # Linux/Mac
dir cleaned_aqi_weather_dataset.csv  # Windows
```

### Step 6: Set Up Hopsworks Feature Store

1. **Prepare data for Hopsworks**:
   ```bash
   python setup_hopsworks.py
   ```
   
   This will:
   - Connect to your Hopsworks project
   - Create a feature group
   - Prepare the data structure

2. **Insert data into Hopsworks**:
   ```bash
   python setup_hopsworks.py --insert
   ```
   
   Or manually insert:
   ```python
   python
   >>> import hopsworks
   >>> import os
   >>> import pandas as pd
   >>> project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'), project=os.getenv('HOPSWORKS_PROJECT_NAME', 'aqi_prediction'))
   >>> fs = project.get_feature_store()
   >>> fg = fs.get_feature_group(name="aqi_weather_features", version=1)
   >>> df = pd.read_csv("cleaned_aqi_weather_dataset.csv")
   >>> df['event_timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
   >>> df['timestamp'] = df['event_timestamp']
   >>> fg.insert(df)  # Automatically commits in newer Hopsworks versions
   >>> exit()
   ```

3. **Feature View is created automatically** by the setup script. No manual step needed!

### Step 7: Train Your Model

Now that data is in Hopsworks, train your model:

**Option A: Using Python Script (Recommended)**
```bash
python improved_model_train.py
```

**Option B: Using Jupyter Notebook**
```bash
jupyter notebook
```
- Open `model_train.ipynb` or `improved_model_train.py`
- Run all cells

**Option C: Using Papermill (Command Line)**
```bash
papermill model_train.ipynb /tmp/model_train_output.ipynb
```

**This will create:**
- `best_model.pkl` - Your trained model
- `best_model_metadata.json` - Model metadata
- `feature_names.json` - Feature names used by the model

**Verify model was created:**
```bash
ls -la best_model.pkl  # Linux/Mac
dir best_model.pkl  # Windows
```

### Step 8: Create the Scaler

The scaler is needed for feature normalization:

```bash
python scripts/create_scaler.py
```

**Verify scaler was created:**
```bash
ls -la scaler.pkl  # Linux/Mac
dir scaler.pkl  # Windows
```

### Step 9: Install Frontend Dependencies

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Return to project root**:
   ```bash
   cd ..
   ```

### Step 10: Verify Setup

Run the verification script:

```bash
python scripts/verify_setup.py
```

This will check:
- ✅ Python packages installed
- ✅ Hopsworks installed and configured
- ✅ Model files exist
- ✅ Scaler exists
- ✅ Frontend dependencies installed

### Step 11: Start the Application

You'll need **two terminals** running:

#### Terminal 1: Start Backend

```bash
# Make sure you're in the project root
cd backend
python main.py
```

You should see:
```
✅ Hopsworks feature store initialized successfully
✅ Model loaded successfully
✅ Scaler loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The API will be available at `http://localhost:8000`

**Alternative (using uvicorn directly):**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2: Start Frontend

```bash
# Make sure you're in the project root
cd frontend
npm run dev
```

You should see:
```
VITE v4.x.x  ready in xxx ms
➜  Local:   http://localhost:5173/
```

**Note:** The frontend might run on port 5173 (Vite default) or 3000, depending on your configuration.

### Step 12: Access the Dashboard

Open your browser and navigate to:

- **Dashboard**: http://localhost:5173 (or http://localhost:3000)
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Root**: http://localhost:8000/

---

## ✅ Verification Checklist

After setup, verify everything is working:

- [ ] Backend starts without errors
- [ ] Frontend loads in browser
- [ ] API health check returns `{"status": "healthy", "hopsworks_available": true}`
- [ ] Forecast endpoint (`/api/forecast`) returns predictions
- [ ] Historical data endpoint (`/api/historical`) returns data
- [ ] Dashboard displays AQI forecasts
- [ ] Dashboard displays historical charts
- [ ] No console errors in browser

---

## 🐛 Troubleshooting

### Hopsworks Connection Issues

**Problem**: "Could not initialize Hopsworks feature store"

**Solutions**:
1. Verify API key is set:
   ```bash
   echo $HOPSWORKS_API_KEY  # Linux/Mac
   echo %HOPSWORKS_API_KEY%  # Windows CMD
   $env:HOPSWORKS_API_KEY  # Windows PowerShell
   ```

2. Test connection manually:
   ```python
   python
   >>> import hopsworks
   >>> import os
   >>> project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'), project=os.getenv('HOPSWORKS_PROJECT_NAME', 'aqi_prediction'))
   >>> print("✅ Connected!")
   ```

3. Check project name matches your Hopsworks project

### Model Not Found Error

**Problem**: Backend shows "Model not loaded" or "Model file not found"

**Solution**:
1. Ensure `best_model.pkl` exists in the project root:
   ```bash
   ls -la best_model.pkl  # Linux/Mac
   dir best_model.pkl  # Windows
   ```
2. If missing, train the model (Step 7)

### Scaler Not Found Error

**Problem**: Backend shows "Scaler not loaded"

**Solution**:
1. Create the scaler:
   ```bash
   python scripts/create_scaler.py
   ```
2. Verify it exists:
   ```bash
   ls -la scaler.pkl  # Linux/Mac
   dir scaler.pkl  # Windows
   ```

### Data File Not Found

**Problem**: Training fails with "Data file not found" or "Failed to retrieve data from Hopsworks"

**Solutions**:
1. **If using Hopsworks**: Ensure data is inserted into feature store (Step 6)
2. **If using CSV fallback**: Ensure `cleaned_aqi_weather_dataset.csv` exists:
   ```bash
   ls -la cleaned_aqi_weather_dataset.csv  # Linux/Mac
   dir cleaned_aqi_weather_dataset.csv  # Windows
   ```
3. If missing, follow Step 5 to generate data

### Port Already in Use

**Problem**: "Address already in use" error

**Solution**:
1. Find and stop the process using the port:

   **Linux/Mac:**
   ```bash
   lsof -ti:8000 | xargs kill -9  # For backend
   lsof -ti:5173 | xargs kill -9  # For frontend
   ```

   **Windows:**
   ```cmd
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. Or change the port in the configuration

### Import Errors

**Problem**: "ModuleNotFoundError" or import errors

**Solution**:
1. Ensure virtual environment is activated:
   ```bash
   which python  # Should point to venv
   ```
2. Reinstall dependencies:
   ```bash
   pip install -r requirements-backend.txt
   ```

### Frontend Build Errors

**Problem**: Frontend won't start or build errors

**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json  # Linux/Mac
rmdir /s node_modules  # Windows
npm install
npm run dev
```

---

## 📊 Quick Reference Commands

### Start Everything
```bash
# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### Update Data in Hopsworks
```bash
python scripts/update_hopsworks.py
```

### Retrain Model
```bash
python improved_model_train.py
```

### Verify Setup
```bash
python scripts/verify_setup.py
```

---

## 🎯 Next Steps

After successful setup:

1. **Explore the Dashboard**: Navigate through different AQI categories and see predictions
2. **Check API Documentation**: Visit http://localhost:8000/docs
3. **Review Predictions**: Check model predictions in the dashboard
4. **Customize**: Modify frontend components or backend endpoints as needed
5. **Set up Automation**: Configure GitHub Actions workflows for automated data updates

---

## 💡 Tips

1. **Use Virtual Environment**: Always use a virtual environment to avoid package conflicts
2. **Check Logs**: If something fails, check the terminal logs for detailed error messages
3. **API Testing**: Use the API docs at http://localhost:8000/docs to test endpoints
4. **Data Updates**: Run the data extraction notebooks periodically to keep data fresh
5. **Model Retraining**: Retrain the model periodically (daily/weekly) for better accuracy
6. **Environment Variables**: Keep your Hopsworks API key secure and don't commit it to git

---

## 🆘 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages in terminal logs
3. Verify all required files exist
4. Ensure all dependencies are installed
5. Check that ports 5173 (or 3000) and 8000 are available
6. Verify Hopsworks connection and API key

---

**Congratulations!** 🎉 You've successfully set up the AQI Prediction Dashboard from scratch!

