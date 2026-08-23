import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def engineer_features(df):
    """Calculate financial ratios used for risk modeling"""
    df = df.copy()
    
    # Avoid division by zero
    revenue_safe = df['annual_revenue'].replace(0, 1)
    
    # Loan-to-Revenue Ratio
    df['loan_to_revenue'] = df['requested_amount'] / revenue_safe
    
    # Debt-to-Revenue Ratio
    df['debt_to_revenue'] = df['existing_debt'] / revenue_safe
    
    # Approximated DSCR (Debt Service Coverage Ratio)
    noi_approx = df['annual_revenue'] * 0.20
    annual_new_debt_service = df['requested_amount'] / (df['term_months'] / 12)
    annual_existing_debt_service = df['existing_debt'] * 0.20
    total_debt_service = annual_existing_debt_service + annual_new_debt_service
    
    # Cap DSCR at 10.0 for extremely safe loans to prevent outlier skewing
    df['dscr'] = np.where(total_debt_service > 0, noi_approx / total_debt_service, 10.0)
    df['dscr'] = np.clip(df['dscr'], 0, 10.0)
    
    return df

def train():
    data_path = 'data/synthetic_loan_data.csv'
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run data_generator.py first.")
        return
        
    print("Loading synthetic data...")
    df = pd.read_csv(data_path)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Define features and target
    features = ['requested_amount', 'term_months', 'annual_revenue', 
                'existing_debt', 'loan_to_revenue', 'debt_to_revenue', 'dscr']
    
    X = df[features]
    y = df['defaulted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature Importances
    print("\n--- Feature Importances ---")
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/risk_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train()
