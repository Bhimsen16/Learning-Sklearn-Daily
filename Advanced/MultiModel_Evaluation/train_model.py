import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Loading customer data ....")
df = pd.read_csv("Customer_analytics.csv")

print("\n---- Checking Feature Means By Churn Status ----")
print(df.groupby('churned').mean(numeric_only=True))

X = df.drop(columns = ["customer_id", "churned"])
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

print(f"Training Shapes: {X_train.shape} | Test Shapes: {X_test.shape}")

# Identify features which needs preprocessing
numeric_features = ['age', 'days_as_member', 'total_spend_npr', 'avg_monthly_visits', 'support_tickets_filed']

preprocessor = ColumnTransformer(
    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')), 
            ('scaler', StandardScaler()), 
        ]), numeric_features),
    ]
)

print("Preprocessing pipeline constructed successfully.")

# Define the models 
models = {
    "Decision Tree (Balanced)": DecisionTreeClassifier(random_state = 42, class_weight = 'balanced'),
    "Random Forest (Balanced)": RandomForestClassifier(random_state = 42, 
                                                       class_weight = 'balanced',
                                                       max_depth=5,
                                                       min_samples_leaf=2),
    "Gradient Boosting (Baseline)": GradientBoostingClassifier(random_state = 42),
}

print("\nStarting the Model Showdown ....")

# Train and evaluate each model using the pipeline defense
for name, model in models.items():
    # Link preprocessor with the specific model inside a clean pipeline
    full_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])
    
    # imputing/scaling happens automatically per fold
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)

    print(f"\n ---- {name} Results ----")
    print(classification_report(y_test, y_pred))

    print(f"---- {name} Confusion Matrix ----")
    print(confusion_matrix(y_test, y_pred))
    print("="*40)