import pandas as pd
import numpy as np
import random
import os
import requests
import io

def generate_synthetic_data(num_samples=5000):
    print("Generating synthetic data...")
    np.random.seed(42)
    random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f"{random.randint(1000, 9999)}-{random.randint(10000, 99999)}" for _ in range(num_samples)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], num_samples)
    senior_citizen = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    partner = np.random.choice(['Yes', 'No'], num_samples)
    dependents = np.random.choice(['Yes', 'No'], num_samples)
    
    # Services
    tenure = np.random.randint(1, 72, num_samples)
    phone_service = np.random.choice(['Yes', 'No'], num_samples, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == 'Yes', np.random.choice(['Yes', 'No', 'No phone service'], num_samples), 'No phone service')
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples, p=[0.4, 0.4, 0.2])
    
    online_security = np.where(internet_service != 'No', np.random.choice(['Yes', 'No'], num_samples), 'No internet service')
    online_backup = np.where(internet_service != 'No', np.random.choice(['Yes', 'No'], num_samples), 'No internet service')
    device_protection = np.where(internet_service != 'No', np.random.choice(['Yes', 'No'], num_samples), 'No internet service')
    tech_support = np.where(internet_service != 'No', np.random.choice(['Yes', 'No'], num_samples), 'No internet service')
    streaming_tv = np.where(internet_service != 'No', np.random.choice(['Yes', 'No'], num_samples), 'No internet service')
    streaming_movies = np.where(internet_service != 'No', np.random.choice(['Yes', 'No'], num_samples), 'No internet service')
    
    # Contract & Billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples, p=[0.5, 0.3, 0.2])
    paperless_billing = np.random.choice(['Yes', 'No'], num_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_samples)
    
    monthly_charges = np.random.uniform(20, 120, num_samples).round(2)
    total_charges = (monthly_charges * tenure).round(2) # Approximate
    
    # Churn Logic
    churn_prob = np.zeros(num_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0.0)
    churn_prob += np.where(contract == 'One year', 0.1, 0.0)
    churn_prob += np.where(internet_service == 'Fiber optic', 0.1, 0.0) 
    churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0.0)
    churn_prob += np.where(tenure < 12, 0.2, 0.0)
    churn_prob -= np.where(partner == 'Yes', 0.05, 0.0)
    churn_prob -= np.where(dependents == 'Yes', 0.05, 0.0)
    churn_prob -= np.where(tech_support == 'Yes', 0.1, 0.0)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    
    churn = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in churn_prob]
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df

def fetch_public_data():
    print("Fetching public Telco Customer Churn dataset...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        print("Public dataset fetched successfully.")
        return df
    except Exception as e:
        print(f"Error fetching public data: {e}")
        return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # 1. Generate Synthetic
    df_synthetic = generate_synthetic_data()
    df_synthetic.to_csv("data/synthetic_churn_data.csv", index=False)
    print("Saved data/synthetic_churn_data.csv")
    
    # 2. Fetch Public
    df_public = fetch_public_data()
    if df_public is not None:
        df_public.to_csv("data/telco_customer_churn.csv", index=False)
        print("Saved data/telco_customer_churn.csv")
        
        # Use public data as the main dataset for the app
        df_public.to_csv("data/customer_churn_data.csv", index=False)
        print("Set public data as main dataset (data/customer_churn_data.csv)")
    else:
        print("Using synthetic data as main dataset.")
        df_synthetic.to_csv("data/customer_churn_data.csv", index=False)
