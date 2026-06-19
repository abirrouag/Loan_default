import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Loan Intelligence Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and metadata
MODEL_PATH = '/home/sary/abir/loan_model.pkl'
METADATA_PATH = '/home/sary/abir/model_metadata.json'

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_assets()

# Inject custom CSS for premium design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #4F46E5 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 400;
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 30px;
    }
    
    /* Card wrapper */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4F46E5;
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.05);
    }
    
    .card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result styling */
    .result-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
    }
    
    .result-denied {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.1);
    }

    .result-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .result-body {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        color: var(--text-color);
        max-width: 600px;
        margin: 0 auto;
    }

    /* Button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
        color: white !important;
        border: none;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    
    div.stButton > button:hover {
        background: linear-gradient(90deg, #4338CA 0%, #4F46E5 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Render main header
st.markdown('<div class="main-title">Loan Intelligence Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time credit scoring and automated loan underwriting engine</div>', unsafe_allow_html=True)

if model is None or metadata is None:
    st.error("⚠️ Model or preprocessing metadata files could not be found! Please run `python train.py` to train the model first.")
    st.stop()

# Sidebar: Model Metrics & Info
st.sidebar.markdown("### 📊 Model Architecture")
st.sidebar.info("""
**Booster:** XGBoost Classifier
**Optimization:** GridSearchCV
**Evaluation Metrics:** F1-Score & Accuracy
""")

# Display validation scores in sidebar
st.sidebar.markdown("### 📈 Training Performance")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("F1 Train Score", f"{metadata['f1_train_score']:.2%}")
col_s2.metric("Test Accuracy", f"{metadata['accuracy_test_score']:.2%}")

st.sidebar.markdown("---")

# Feature importance view in sidebar
st.sidebar.markdown("### 🔍 Top Predictive Features")
# We know from feature_importances: total_income, CoapplicantIncome, ApplicantIncome, LoanAmount, dti_ratio, Dependents
importances = {
    "Total Income": 0.228,
    "Coapplicant Income": 0.204,
    "Applicant Income": 0.198,
    "Loan Amount": 0.195,
    "Debt-to-Income": 0.156,
    "Dependents": 0.018
}
st.sidebar.bar_chart(pd.Series(importances))

st.sidebar.markdown("⚠️ *Note: Categorical text variables (Married, Gender, Education, Self_Employed, Property_Area) were coerced to 0 in the notebook's preprocessing pipeline and thus carry 0 weight in this model.*")

# Main Application Form
st.markdown("### 📝 Enter Applicant Details")

# Form layouts
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-card"><div class="card-title">👤 Applicant Profile</div>', unsafe_allow_html=True)
    dependents = st.selectbox(
        "Number of Dependents",
        options=["0", "1", "2", "3+"],
        index=0,
        help="Select number of dependents. '3+' maps to 0 based on preprocessing rules."
    )
    
    credit_history = st.selectbox(
        "Credit History",
        options=["Good Credit (No Defaults)", "Poor Credit (Past Defaults)"],
        index=0,
        help="Select the applicant's credit history status."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-card"><div class="card-title">💰 Financial Profile</div>', unsafe_allow_html=True)
    applicant_income = st.number_input(
        "Applicant Monthly Income ($)",
        min_value=0,
        value=5000,
        step=500,
        help="Main applicant's monthly income."
    )
    
    coapplicant_income = st.number_input(
        "Coapplicant Monthly Income ($)",
        min_value=0,
        value=1500,
        step=250,
        help="Coapplicant's monthly income (enter 0 if none)."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card"><div class="card-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    loan_amount = st.number_input(
        "Requested Loan Amount ($ in thousands)",
        min_value=1,
        value=130,
        step=10,
        help="The amount of the loan in thousands (e.g. 130 means $130,000)."
    )
    
    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=12,
        max_value=480,
        value=360,
        step=12,
        help="Duration of the loan. Preprocessing clips this to 360 months."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-card"><div class="card-title">ℹ️ Preprocessing & Clipping Insights</div>', unsafe_allow_html=True)
    # Interactive explanation of what happens to their inputs
    total_inc = applicant_income + coapplicant_income
    dti = loan_amount / (applicant_income + 1)
    st.write(f"- **Calculated Total Income:** ${total_inc:,.2f}")
    st.write(f"- **Calculated Debt-to-Income:** {dti:.4f}")
    
    # Clip indicators
    bounds = metadata['clipping_bounds']
    is_inc_clipped = applicant_income > bounds['ApplicantIncome']['upper']
    is_amt_clipped = loan_amount > bounds['LoanAmount']['upper']
    
    if is_inc_clipped:
        st.warning(f"⚠️ Applicant Income will be clipped to ${bounds['ApplicantIncome']['upper']:,.0f} (outlier limit).")
    if is_amt_clipped:
        st.warning(f"⚠️ Loan Amount will be clipped to ${bounds['LoanAmount']['upper']:,.0f}k (outlier limit).")
    st.markdown('</div>', unsafe_allow_html=True)

# Submit Button
if st.button("Evaluate Loan Application"):
    with st.spinner("Processing underwriting pipeline..."):
        # Map credit history to float
        ch_val = 1.0 if "Good" in credit_history else 0.0
        
        # 1. Base values for standard features (from metadata/preproc)
        gender = 0.0
        married = 0.0
        education = 0.0
        self_employed = 0.0
        property_area = 0.0
        loan_status_placeholder = 0.0
        
        # 2. Dependents mapping
        dep_val = 0.0 if dependents == '3+' else float(dependents)
        
        # 3. Clip inputs using training bounds
        bounds = metadata['clipping_bounds']
        app_inc_clipped = np.clip(float(applicant_income), bounds['ApplicantIncome']['lower'], bounds['ApplicantIncome']['upper'])
        coapp_inc_clipped = np.clip(float(coapplicant_income), bounds['CoapplicantIncome']['lower'], bounds['CoapplicantIncome']['upper'])
        loan_amt_clipped = np.clip(float(loan_amount), bounds['LoanAmount']['lower'], bounds['LoanAmount']['upper'])
        loan_term_clipped = np.clip(float(loan_term), bounds['Loan_Amount_Term']['lower'], bounds['Loan_Amount_Term']['upper'])
        credit_hist_clipped = np.clip(float(ch_val), bounds['Credit_History']['lower'], bounds['Credit_History']['upper'])
        
        # 4. Feature engineering
        dti_ratio = loan_amt_clipped / (app_inc_clipped + 1)
        total_income = app_inc_clipped + coapp_inc_clipped
        loan_to_income = loan_amt_clipped / (app_inc_clipped + 1)
        
        # Bin mapping using numpy digitize
        ti_bins = metadata['total_income_bins']
        la_bins = metadata['loan_amount_bins']
        
        income_category = int(np.digitize(total_income, ti_bins[1:-1]))
        loan_category = int(np.digitize(loan_amt_clipped, la_bins[1:-1]))
        
        # 5. Construct DataFrame with correct features and order
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dep_val],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [app_inc_clipped],
            'CoapplicantIncome': [coapp_inc_clipped],
            'LoanAmount': [loan_amt_clipped],
            'Loan_Amount_Term': [loan_term_clipped],
            'Credit_History': [credit_hist_clipped],
            'Property_Area': [property_area],
            'Loan_Status': [loan_status_placeholder],
            'dti_ratio': [dti_ratio],
            'total_income': [total_income],
            'loan_to_income': [loan_to_income],
            'income_category': [income_category],
            'loan_category': [loan_category]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities[prediction]
        
        # Display Result
        if prediction == 1:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-header">✅ Loan Approved</div>
                <div class="result-body">
                    The model predicts that this loan application is <strong>eligible</strong> for approval.
                    <br>
                    <strong>Underwriting Confidence:</strong> {confidence:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="result-denied">
                <div class="result-header">❌ Loan Denied</div>
                <div class="result-body">
                    The model predicts that this loan application is <strong>not eligible</strong> for approval.
                    <br>
                    <strong>Underwriting Confidence:</strong> {confidence:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Show feature details in a beautiful table
        st.markdown("### 📊 Processed Underwriting Features")
        display_df = input_data.T.reset_index()
        display_df.columns = ['Feature Name', 'Processed Value Fed to Model']
        st.dataframe(display_df, use_container_width=True)
