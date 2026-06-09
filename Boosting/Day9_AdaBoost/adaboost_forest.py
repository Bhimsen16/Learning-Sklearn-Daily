import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load the dataset and split it 70/30
wine = load_wine()
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Initialize the model and train it
forest = RandomForestClassifier(n_estimators = 100, max_depth = 1, random_state = 42)
adaboost_forest = AdaBoostClassifier(n_estimators = 100, random_state = 42)

forest.fit(X_train, y_train)
adaboost_forest.fit(X_train, y_train)

# Print the test set accuracy
forest_acc = accuracy_score(y_test, forest.predict(X_test))
adaboost_acc = accuracy_score(y_test, adaboost_forest.predict(X_test))

print(f"Random Forest Accuracy with max depth 1: {forest_acc * 100:.2f}%")
print(f"AdaBoost Forest Accuracy with default max depth 1: {adaboost_acc * 100:.2f}%")
