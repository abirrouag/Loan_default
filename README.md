# Loan Intelligence Portal (Machine Learning End-to-End)

Welcome to the **Loan Underwriting and Intelligence Portal**, an end-to-end Machine Learning solution designed to predict loan application eligibility in real-time. This project handles the entire data lifecycle, from raw data exploration and cleaning to model optimization and interactive local deployment.

---

## 🚀 Quick Start: Running the Streamlit App

A shell script is provided to automate environment verification and run the application locally.

### 1. Execute the Run Script
In your terminal, navigate to the project directory and execute:
```bash
./run_app.sh
```

### 2. Manual Alternative
If you prefer to run it manually or need to install dependencies individually:
```bash
# 1. Install required packages
pip install streamlit pandas numpy xgboost scikit-learn joblib

# 2. Run the Streamlit application
streamlit run app.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 📁 Repository Structure

*   **`train_u6lujuX_CVtuZ9i.csv`**: The raw training dataset containing demographic and financial profiles of applicants.
*   **`loan-ml (1).ipynb`**: The original research and experimentation notebook detailing the exploratory data analysis (EDA) and model development.
*   **`train.py`**: Cleaned, reproducible training pipeline that reads the dataset, cleans data, performs feature engineering, runs GridSearchCV, and serializes the assets.
*   **`loan_model.pkl`**: The serialized, optimized XGBoost Classifier model.
*   **`model_metadata.json`**: Preprocessing metadata (outlier bounds, median statistics, quantile bin edges) to ensure perfect feature compatibility during inference.
*   **`app.py`**: Streamlit application script containing the frontend UI, user input components, and the model prediction pipeline.
*   **`run_app.sh`**: Helper shell script to bootstrap dependencies and start the local Streamlit server.

---

## ⚙️ Preprocessing & Feature Engineering Pipeline

During inference, user input is processed using the exact same metrics and bounds calculated from the training dataset to prevent data leakage and guarantee prediction alignment:

1.  **Imputation:** Missing numerical values are filled using training medians (e.g., `ApplicantIncome` median: `$3,812.50`, `LoanAmount` median: `$128,000`).
2.  **Outlier Capping:** Input values are clipped using the training set's Interquartile Range (IQR) limits:
    *   `ApplicantIncome` is clipped to `[-1,498.75, 10,171.25]`
    *   `CoapplicantIncome` is clipped to `[-3,445.88, 5,743.13]`
    *   `LoanAmount` is clipped to `[3.50, 261.50]`
3.  **Feature Engineering:**
    *   **Debt-to-Income (DTI) Ratio:** $\text{LoanAmount} / (\text{ApplicantIncome} + 1)$
    *   **Total Income:** $\text{ApplicantIncome} + \text{CoapplicantIncome}$
    *   **Loan-to-Income Ratio:** $\text{LoanAmount} / (\text{ApplicantIncome} + 1)$
    *   **Quantile Binning:** Total Income and Loan Amount are categorized into 4 ordinal groups based on the training quantiles and encoded to codes `0`, `1`, `2`, or `3`.
4.  **Categorical Alignment:** Categorical features that were coerced to numeric codes or 0 in the notebook's training code (e.g., `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`) are kept in the feature list as constant markers to maintain model compatibility without introducing noise.

---

## 📈 Model Performance & Parameters

The predictive model is built using an optimized **XGBoost Classifier** tuned via **GridSearchCV** with 3-fold cross-validation targeting **F1-Score** (to account for class imbalance between approved and denied loans):

*   **Best Parameters:**
    *   `max_depth`: `3`
    *   `learning_rate`: `0.01`
    *   `scale_pos_weight`: `2.0` (weight adjustment for minority class defaults)
*   **Model Scores:**
    *   **F1 Training Score:** `74.44%`
    *   **Test Accuracy:** `62.60%`

### Feature Importance Breakdown
Based on XGBoost booster feature importance:
1.  **Total Income** (~22.8% importance)
2.  **Coapplicant Income** (~20.4% importance)
3.  **Applicant Income** (~19.8% importance)
4.  **Loan Amount** (~19.5% importance)
5.  **Debt-to-Income Ratio** (~15.6% importance)
6.  **Dependents** (~1.8% importance)
