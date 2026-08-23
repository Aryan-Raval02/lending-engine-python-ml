import os
import joblib
import pandas as pd
import numpy as np

# Try to load the model on startup
MODEL_PATH = 'models/risk_model.pkl'
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Risk model loaded from {MODEL_PATH}")
    else:
        model = None
        print(f"Warning: Model not found at {MODEL_PATH}. Will use fallback heuristic rules.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def determine_category(score):
    if score >= 80:
        return "LOW"
    elif score >= 60:
        return "MEDIUM"
    elif score >= 40:
        return "HIGH"
    else:
        return "VERY_HIGH"

def evaluate_risk(application_data: dict) -> dict:
    """
    Evaluates loan application risk.
    If ML model is loaded, uses it. Otherwise uses fallback DSCR rules.
    """
    req_amount = application_data.get("requested_amount", 0)
    term = application_data.get("term_months", 12)
    revenue = application_data.get("annual_revenue", 0)
    debt = application_data.get("existing_debt", 0)
    
    # Engineer features
    revenue_safe = max(revenue, 1)
    loan_to_revenue = req_amount / revenue_safe
    debt_to_revenue = debt / revenue_safe
    
    noi_approx = revenue * 0.20
    annual_new_debt_service = req_amount / max(term / 12, 0.1)
    annual_existing_debt_service = debt * 0.20
    total_debt_service = annual_existing_debt_service + annual_new_debt_service
    
    dscr = noi_approx / total_debt_service if total_debt_service > 0 else 10.0
    dscr = min(max(dscr, 0), 10.0)
    
    features = {
        'requested_amount': req_amount,
        'term_months': term,
        'annual_revenue': revenue,
        'existing_debt': debt,
        'loan_to_revenue': loan_to_revenue,
        'debt_to_revenue': debt_to_revenue,
        'dscr': dscr
    }
    
    if model is not None:
        # ML Evaluation
        df = pd.DataFrame([features])
        # Model returns probability of default (class 1)
        prob_default = model.predict_proba(df)[0][1]
        
        # Convert prob_default to a 0-100 score where 100 is best (safest)
        # 0% default -> 100 score. 100% default -> 0 score.
        score_val = (1.0 - prob_default) * 100
        score = round(score_val, 2)
        
        # Calculate feature contributions (basic heuristic based on global importances)
        # For true SHAP values, we'd need shap library, but this is a lightweight approximation
        importances = model.feature_importances_
        feature_names = df.columns
        contributions = {name: round(imp * 100, 1) for name, imp in zip(feature_names, importances)}
        
    else:
        # Fallback DSCR Heuristic Evaluation
        print("Using fallback heuristic evaluation (no ML model)")
        
        # Base score 100
        penalty = 0
        
        # DSCR penalty
        if dscr < 1.0:
            penalty += 50
        elif dscr < 1.25:
            penalty += 30
        elif dscr < 1.5:
            penalty += 10
            
        # Debt/Revenue penalty
        if debt_to_revenue > 0.8:
            penalty += 30
        elif debt_to_revenue > 0.5:
            penalty += 15
            
        score = max(100 - penalty, 0)
        
        contributions = {
            "dscr_impact": "High" if dscr < 1.25 else "Low",
            "debt_impact": "High" if debt_to_revenue > 0.8 else "Low"
        }

    category = determine_category(score)
    
    # Generate human readable reasons
    reasons = []
    if dscr < 1.25:
        reasons.append(f"Low Debt Service Coverage Ratio ({dscr:.2f}) indicates potential cash flow stress.")
    if debt_to_revenue > 0.6:
        reasons.append(f"High existing debt relative to revenue ({(debt_to_revenue*100):.1f}%).")
    if loan_to_revenue > 0.5:
        reasons.append(f"Requested loan amount is very high compared to annual revenue.")
        
    if not reasons and score >= 80:
        reasons.append("Strong financials with healthy coverage ratios.")
        
    return {
        "score": score,
        "category": category,
        "reasons": reasons,
        "metrics": {
            "dscr": round(dscr, 2),
            "debt_to_revenue_ratio": round(debt_to_revenue, 3),
            "loan_to_revenue_ratio": round(loan_to_revenue, 3)
        },
        "feature_importances": contributions
    }
