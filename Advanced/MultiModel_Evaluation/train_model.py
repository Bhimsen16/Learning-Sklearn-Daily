import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline

print("Loading customer data ....")
df = pd.read_csv("Customer_analytics.csv")

X = df.drop(columns = ["customer_id", "churned"])
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

print(f"Training Shapes: {X_train.shape} Test Shapes: {X_test.shape}")

# Identify features which needs preprocessing
numeric_features = ['age', 'days_as_member', 'total_spend_usd', 'avg_monthly_visits', 'support_tickets_filed']

preprocessor = ColumnTransformer(
    transformers = [
        "num", Pipeline([
            ("imputer", SimpleImputer(strategy='median')), 
            ("scaler", StandardScaler()), 
        ])
    ]
)

print("Preprocessing pipeline constructed successfully.")