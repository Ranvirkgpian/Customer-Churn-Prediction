import streamlit as st
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    if os.path.exists("data/customer_churn_data.csv"):
        return pd.read_csv("data/customer_churn_data.csv")
    return None

df = load_data()

# Load Model & Artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("models/best_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        categorical_values = joblib.load("models/categorical_values.pkl")
        return model, feature_names, categorical_values
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, feature_names, categorical_values = load_model_artifacts()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Churn Predictor", "High Risk Customers", "SQL Playground", "Strategies Report"])

# --- PAGE: Dashboard Overview ---
if page == "Dashboard Overview":
    st.title("📉 Customer Churn Dashboard")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        total_customers = len(df)
        churn_count = df[df['Churn'] == 'Yes'].shape[0]
        churn_rate = (churn_count / total_customers) * 100
        
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churn Rate", f"{churn_rate:.2f}%")
        col3.metric("Total Revenue (Est)", f"${df['TotalCharges'].sum():,.2f}")
        
        st.markdown("### Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
        
        st.markdown("### Churn by Contract Type")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='Contract', hue='Churn', ax=ax2)
        st.pyplot(fig2)
        
    else:
        st.error("Data not found. Please ensure data is generated.")

# --- PAGE: Churn Predictor ---
elif page == "Churn Predictor":
    st.title("🔮 Churn Probability Predictor")
    st.markdown("Enter customer details to predict churn probability.")
    
    if model and feature_names and categorical_values:
        input_data = {}
        
        # Create form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            # Numeric Inputs
            with col1:
                tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=monthly_charges*tenure)
                
            # Categorical Inputs
            with col2:
                # We need to reconstruct the input dataframe exactly as the model expects
                # We loop through feature_names and create inputs
                # But we need to handle them nicely (not just a big loop)
                pass

            # Let's manually map the important ones for better UI, or loop if lazy.
            # To be robust, let's loop but organize.
            
            # Helper to get options
            def get_options(col):
                return categorical_values.get(col, [])

            # Categorical fields (excluding the ones handled above)
            cat_cols = [c for c in feature_names if c not in ['tenure', 'MonthlyCharges', 'TotalCharges']]
            
            # Split categorical into two columns for layout
            mid = len(cat_cols) // 2
            
            with col1:
                for col in cat_cols[:mid]:
                    input_data[col] = st.selectbox(col, get_options(col), key=col)
            
            with col2:
                for col in cat_cols[mid:]:
                    input_data[col] = st.selectbox(col, get_options(col), key=col)
            
            # Add numeric to input_data
            input_data['tenure'] = tenure
            input_data['MonthlyCharges'] = monthly_charges
            input_data['TotalCharges'] = total_charges
            
            submit = st.form_submit_button("Predict Churn")
            
        if submit:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure column order matches training
            input_df = input_df[feature_names]
            
            # Predict
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                if prediction == 1:
                    st.error(f"⚠️ High Risk! Churn Probability: {probability:.2%}")
                else:
                    st.success(f"✅ Low Risk. Churn Probability: {probability:.2%}")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.error("Model not loaded.")

# --- PAGE: High Risk Customers ---
elif page == "High Risk Customers":
    st.title("🚨 High Risk Customers")
    
    if df is not None and model:
        st.write("Scoring all customers...")
        
        # Prepare data for scoring
        X = df.drop(['customerID', 'Churn'], axis=1)
        # Ensure order
        X = X[feature_names]
        
        # Predict Proba
        probs = model.predict_proba(X)[:, 1]
        
        df_scored = df.copy()
        df_scored['Churn_Probability'] = probs
        
        # Filter high risk
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7)
        high_risk_df = df_scored[df_scored['Churn_Probability'] >= threshold].sort_values(by='Churn_Probability', ascending=False)
        
        st.write(f"Found {len(high_risk_df)} customers with churn probability >= {threshold}")
        
        st.dataframe(high_risk_df[['customerID', 'Churn_Probability', 'Contract', 'MonthlyCharges', 'tenure']])
        
        # Download button
        csv = high_risk_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download High Risk List (CSV)",
            csv,
            "high_risk_customers.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.error("Data or Model not available.")

# --- PAGE: SQL Playground ---
elif page == "SQL Playground":
    st.title("💾 SQL Data Playground")
    st.markdown("Write SQL queries to analyze the data. The table name is `customers`.")
    
    if df is not None:
        # Setup in-memory DB
        conn = sqlite3.connect(":memory:")
        df.to_sql("customers", conn, index=False, if_exists="replace")
        
        default_query = "SELECT Contract, AVG(MonthlyCharges) as AvgPrice, COUNT(*) as Count FROM customers GROUP BY Contract"
        query = st.text_area("SQL Query", value=default_query, height=150)
        
        if st.button("Run Query"):
            try:
                result = pd.read_sql_query(query, conn)
                st.dataframe(result)
            except Exception as e:
                st.error(f"SQL Error: {e}")
    else:
        st.error("Data not loaded.")

# --- PAGE: Strategies Report ---
elif page == "Strategies Report":
    st.title("📋 Retention Strategies & Report")
    
    if os.path.exists("REPORT.md"):
        with open("REPORT.md", "r") as f:
            report_content = f.read()
        st.markdown(report_content)
    else:
        st.error("Report file not found.")
