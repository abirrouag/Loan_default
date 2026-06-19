import os
import json
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
import joblib

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'train_u6lujuX_CVtuZ9i.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    # 1. Map target
    target_column = 'Loan_Status'
    df['loan_status'] = df[target_column].map({'Y': 1, 'N': 0})

    # 2. Drop ID columns
    for col in ['Loan_ID', 'id', 'ID', 'customer_id']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # 3. Separate column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'loan_status' in numeric_cols:
        numeric_cols.remove('loan_status')
    if 'loan_status' in categorical_cols:
        categorical_cols.remove('loan_status')

    # 4. Handle missing values
    medians = {}
    for col in numeric_cols:
        if col in df.columns:
            median_val = float(df[col].median())
            medians[col] = median_val
            df[col] = df[col].fillna(median_val)
            
    modes = {}
    for col in categorical_cols:
        if col in df.columns:
            mode_val = str(df[col].mode()[0])
            modes[col] = mode_val
            df[col] = df[col].fillna(mode_val)

    # 5. Handle outliers using IQR method (calculate and save clipping bounds)
    clipping_bounds = {}
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = float(Q1 - 1.5 * IQR)
            upper_bound = float(Q3 + 1.5 * IQR)
            clipping_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
            df[col] = df[col].clip(lower_bound, upper_bound)

    # 6. Clean categorical columns (whitespace strip)
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()

    # 7. Feature engineering
    df['dti_ratio'] = df['LoanAmount'] / (df['ApplicantIncome'] + 1)
    df['total_income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['loan_to_income'] = df['LoanAmount'] / (df['ApplicantIncome'] + 1)

    # Cut bounds for categories
    _, total_income_bins = pd.qcut(df['total_income'], q=4, retbins=True, labels=False)
    _, loan_amount_bins = pd.qcut(df['LoanAmount'], q=4, retbins=True, labels=False)
    
    df['income_category'] = pd.qcut(df['total_income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    df['loan_category'] = pd.qcut(df['LoanAmount'], q=4, labels=['Small', 'Medium', 'Large', 'Very Large'])

    # 8. Encode categorical columns (Cell 49 style)
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.codes

    # Force conversion of everything to float to match the notebook's training behavior
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    # 9. Split features and target
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)

    # Split for Grid Search
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 10. Balance scale ratio
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

    # 11. Train model via Grid Search
    print("Training XGBoost Classifier via Grid Search...")
    grid_search = GridSearchCV(
        estimator=XGBClassifier(eval_metric='logloss', scale_pos_weight=ratio),
        param_grid={'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Calculate train & test scores
    train_score = grid_search.best_score_
    test_score = best_model.score(X_test, y_test)
    print(f"Model trained successfully. F1 Train Score: {train_score:.4f}, Accuracy Test Score: {test_score:.4f}")

    # 12. Save model
    model_output_path = os.path.join(base_dir, 'loan_model.pkl')
    joblib.dump(best_model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # 13. Save preprocessing metadata for the streamlit app
    metadata = {
        'features_list': X.columns.tolist(),
        'medians': medians,
        'modes': modes,
        'clipping_bounds': clipping_bounds,
        'total_income_bins': total_income_bins.tolist(),
        'loan_amount_bins': loan_amount_bins.tolist(),
        'f1_train_score': float(train_score),
        'accuracy_test_score': float(test_score)
    }
    
    metadata_path = os.path.join(base_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")

if __name__ == '__main__':
    train_model()
