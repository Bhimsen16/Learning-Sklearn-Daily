import pandas as pd
import numpy as np

np.random.seed(42)
n_customers = 1000

print("Generating synthetic E-commerce Analytics dataset")

# Core customer features
age = np.random.randint(18, 65, size = n_customers)
days_as_member = np.random.randint(10, 1000, size = n_customers)
total_spend = np.round(np.random.exponential(scale=500, size=n_customers) + (days_as_member * 0.5), 2)
avg_monthly_visits = np.random.randint(1, 30, size = n_customers)

# Introduce some missing values in support_tickets
support_tickets = np.random.poisson(lam=1.5, size=n_customers).astype(float)
mask = np.random.rand(n_customers) < 0.1  # 10% missing data
support_tickets[mask] = np.nan

# A hidden pattern for Churn (The Target Variable)
# Churn risk increases if spend is low, visits are low, or support tickets are high
churn_probability = 1 / (1 + np.exp(-(
    0.05 * (40 - age) + 
    -0.003 * days_as_member + 
    -0.002 * total_spend + 
    -0.15 * avg_monthly_visits + 
    0.6 * np.nan_to_num(support_tickets, nan=1.5)
)))
churn = (np.random.rand(n_customers) < churn_probability).astype(int)

df = pd.DataFrame({
    "customer_id": range(1001, 1001 + n_customers),
    "age": age,
    "days_as_member": days_as_member,
    "total_spend_npr": total_spend, 
    "avg_monthly_visits": avg_monthly_visits, 
    "support_tickets_filed": support_tickets,
    "churned": churn,
})

df.to_csv("Customer_analytics.csv", index = False)
print("Saving to CSV file....")
print("Done! 'Customer_analytics_csv' has been generated with 1000 records.")