import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Loading ecommerce customer data ....")
df = pd.read_csv("Customer_analytics.csv")

X = df.drop(columns=['customer_id', 'churned'])
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

print(f"Data split complete. Training shapes: {X_train.shape}, Test shapes: {X_test.shape}")

# Identify features that need preprocessing
# In our data, 'support_tickets_filed' has missing values
numeric_features = ['age', 'days_as_member', 'total_spend_usd', 'avg_monthly_visits', 'support_tickets_filed']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # Fixes missing ticket numbers safely
            ('scaler', StandardScaler()),                  # Balances scales of USD vs counts
        ]), numeric_features)
    ]
)

print("Preprocessing pipeline constructed successfully.")