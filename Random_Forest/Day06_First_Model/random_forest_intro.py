# Import the necessary tools
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load both algorithms to compare them
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load wine dataset
wine = load_wine()
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

# Split the data into 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Training the models
print("--- Training the Models ---")

# Model 1: A single Decision Tree
model_tree = DecisionTreeClassifier(max_depth = 3, random_state = 42)
model_tree.fit(X_train, y_train)

tree_pred = model_tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)

# Model 2: The Random Forest
model_forest = RandomForestClassifier(n_estimators = 100, random_state = 42) #n_estimators = 100 is a committeee of 100 trees inside the forest
model_forest.fit(X_train, y_train)

forest_pred = model_forest.predict(X_test)
forest_accuracy = accuracy_score(y_test, forest_pred)
print("--- Models Trained! ---")

# The results 
print("\n--- Results ---")
print(f"Single Decision Tree Accuracy: {tree_accuracy * 100:.1f}%")
print(f"Random Forest with 100 Trees Accuracy: {forest_accuracy * 100:.1f}%")