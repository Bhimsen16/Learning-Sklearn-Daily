import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Import pipeline tool 
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

dataset = load_breast_cancer()
X = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
y = dataset.target

# Split the data before any preprocessing happens
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Create a Pipeline (To prevent any leakages)
# Define a list of tuples: "name of steps", object_to_run
leak_proof_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Scale features safely
    ("classifier", RandomForestClassifier(random_state = 42))
])

leak_proof_pipeline.fit(X_train, y_train)

y_pred = leak_proof_pipeline.predict(X_test)

print("--- Pipeline Report ---")
print(classification_report(y_test, y_pred, target_names=dataset.target_names))