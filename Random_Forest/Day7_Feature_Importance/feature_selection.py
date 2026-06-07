import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
wine = load_wine()

# Split the dataset into features and labels (needed to calculate accuracy)
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Train the model
model = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 42)
model.fit(X_train, y_train)

model_acc = accuracy_score(y_test, model.predict(X_test))

# Get features that score Above 0.05 (5%) importance based on training data
importance = model.feature_importances_
imp_columns = X.columns[importance > 0.05]

print("--- Feature selection summary ---")
print(f"Original number of features: {len(X.columns)}")
print(f"Reduced number of features (Importance > 5%): {len(imp_columns)}")
print(f"Features Kept: {list(imp_columns)}\n")

# Train a new lean forest with only those top features
X_train_lean = X_train[imp_columns]
X_test_lean = X_test[imp_columns]

feat_model = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 42)
feat_model.fit(X_train_lean, y_train)

feat_model_acc = accuracy_score(y_test, feat_model.predict(X_test_lean))

# Print the accuracy of both models
print(f"Full Forest Accuracy (All 13 Features): {model_acc * 100:.2f}%")
print(f"Lean Forest Accuracy ({len(imp_columns)} Features): {feat_model_acc * 100:.2f}%")