import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def train_model():
    print("Loading data...")
    df = pd.read_csv("data/customer_churn_data.csv")
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Clean TotalCharges (handle empty strings which occur in real data)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Define features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Identify column types
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    print("\nTraining models...")
    for name, model in models.items():
        # Create full pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl") # Save target encoder just in case
    
    # Save feature names for later use in app (to recreate input form)
    # We save the column names of X to know what input is expected
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    
    # Also save unique values for categorical columns to populate dropdowns in Streamlit
    categorical_values = {col: df[col].unique().tolist() for col in categorical_features}
    joblib.dump(categorical_values, "models/categorical_values.pkl")
    
    print("Model and artifacts saved to models/")

if __name__ == "__main__":
    train_model()
