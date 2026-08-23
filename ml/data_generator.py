import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_samples=10000):
    print(f"Generating {num_samples} synthetic loan records...")
    
    # 1. Loan Amount (Requested Amount): uniformly distributed between 10k and 1M
    requested_amount = np.random.uniform(10000, 1000000, num_samples).round(2)
    
    # 2. Term Months: discrete options typical in B2B lending
    term_months = np.random.choice([6, 12, 24, 36, 48, 60], num_samples)
    
    # 3. Annual Revenue: somewhat correlated to requested amount (businesses usually ask for < 50% revenue)
    # Added some noise for realism
    revenue_multiplier = np.random.uniform(1.5, 5.0, num_samples)
    annual_revenue = (requested_amount * revenue_multiplier).round(2)
    
    # Add a few startups with low revenue but asking for high loans
    startup_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
    annual_revenue[startup_indices] = (requested_amount[startup_indices] * np.random.uniform(0.1, 0.8, len(startup_indices))).round(2)
    
    # 4. Existing Debt: somewhat correlated to revenue
    debt_ratio = np.random.uniform(0.0, 0.8, num_samples)
    existing_debt = (annual_revenue * debt_ratio).round(2)
    
    # Add high-debt companies
    high_debt_indices = np.random.choice(num_samples, int(num_samples * 0.1), replace=False)
    existing_debt[high_debt_indices] = (annual_revenue[high_debt_indices] * np.random.uniform(0.8, 1.5, len(high_debt_indices))).round(2)
    
    # 5. Calculate DSCR (Debt Service Coverage Ratio) surrogate for training
    # DSCR = Net Operating Income / Total Debt Service
    # Here we approximate NOI as 20% of revenue, and Debt Service as existing debt + new loan / term
    noi_approx = annual_revenue * 0.20
    annual_new_debt_service = requested_amount / (term_months / 12)
    annual_existing_debt_service = existing_debt * 0.20 # assume 20% of existing debt is paid annually
    total_debt_service = annual_existing_debt_service + annual_new_debt_service
    
    dscr = np.where(total_debt_service > 0, noi_approx / total_debt_service, 5.0)
    
    # 6. Default Logic (Target Variable)
    # Higher chance of default if DSCR < 1.25, or Debt/Revenue > 0.8
    prob_default = np.zeros(num_samples)
    prob_default += np.where(dscr < 1.0, 0.40, 0)
    prob_default += np.where((dscr >= 1.0) & (dscr < 1.25), 0.15, 0)
    prob_default += np.where((existing_debt / annual_revenue) > 0.8, 0.25, 0)
    prob_default += np.where((existing_debt / annual_revenue) > 1.2, 0.20, 0) # cumulative
    
    # Baseline default risk
    prob_default += 0.02 
    
    # Cap probability at 0.95
    prob_default = np.clip(prob_default, 0.01, 0.95)
    
    # Generate actual default outcomes based on probabilities
    defaulted = np.random.binomial(1, prob_default)
    
    # Create DataFrame
    df = pd.DataFrame({
        'requested_amount': requested_amount,
        'term_months': term_months,
        'annual_revenue': annual_revenue,
        'existing_debt': existing_debt,
        'defaulted': defaulted
    })
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    file_path = 'data/synthetic_loan_data.csv'
    df.to_csv(file_path, index=False)
    print(f"Data generation complete. Defaults: {df['defaulted'].mean()*100:.1f}%")
    print(f"Saved to {file_path}")
    
if __name__ == "__main__":
    generate_synthetic_data()
