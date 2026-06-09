import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load the dataset and split it 70/30
wine = load_wine()
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Configurations A: The Agressive Standard (100 estimators and full learning rate)
standard_model = AdaBoostClassifier(n_estimators = 100, learning_rate = 1.0, random_state = 42)
standard_model.fit(X_train, y_train)
s_model_accuracy = accuracy_score(y_test, standard_model.predict(X_test))

# Configuration B: The Cautious Strategiest (50 estimators, smaller learning rate)
cautious_model = AdaBoostClassifier(n_estimators = 50, learning_rate = 0.5, random_state = 42)
cautious_model.fit(X_train, y_train)
c_model_accuracy = accuracy_score(y_test, cautious_model.predict(X_test))

# Print the Comparision Results
print("--- ADABoost HyperParameter Test ---")
print(f"Config A(100 estimators, Learning rate = 1.0) Accuracy: {s_model_accuracy * 100:.2f}%")
print(f"Config B(50 estimators = 50, Learning rate = 0.5) Accuracy: {c_model_accuracy * 100:.2f}%")