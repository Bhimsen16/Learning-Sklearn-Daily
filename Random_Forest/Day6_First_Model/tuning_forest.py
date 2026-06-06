import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and split the dataset
wine = load_wine()
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

print("--- Running Hyperparameter Experiment ---")

# Model A: Small Forest with 10 trees only, shallow depth
model_a = RandomForestClassifier(n_estimators = 10, max_depth = 2, random_state = 42)
model_a.fit(X_train, y_train)
acc_a = accuracy_score(y_test, model_a.predict(X_test))
print(f"Model A accuracy(10 trees, max_depth = 2): {acc_a * 100:.1f}%")

# Model B: Large Forest with 200 Trees, deeper depth
model_b = RandomForestClassifier(n_estimators = 200, max_depth = 5, random_state = 42)
model_b.fit(X_train, y_train)
acc_b = accuracy_score(y_test, model_b.predict(X_test))
print(f"Model B accuracy (200 trees, max_depth = 5): {acc_b * 100:.1f}%")

# Model C: Custom feature constraints
# log2 = limits features mathematically to ensure extreme tree diversity
model_c = RandomForestClassifier(n_estimators = 100, max_features = "log2", random_state = 42)
model_c.fit(X_train, y_train)
acc_c = accuracy_score(y_test, model_c.predict(X_test))
print(f"Model C accuracy (100 trees, max_features = 'log2'): {acc_c * 100:.1f}%")
