#!/bin/bash
# Script to run the local Streamlit application

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit is not installed. Installing required dependencies..."
    pip install streamlit pandas numpy xgboost scikit-learn joblib
fi

# Run the app
echo "Starting Streamlit App..."
streamlit run /home/sary/abir/app.py
