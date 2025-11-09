#!/bin/bash

echo "========================================"
echo "Training Improved AQI Prediction Model"
echo "========================================"
echo ""

python improved_model_train.py

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "The new model has been saved to:"
echo "  - best_model.pkl"
echo "  - scaler.pkl"
echo "  - feature_names.json"
echo "  - best_model_metadata.json"
echo ""
echo "Please restart your backend server to use the new model."
echo ""

