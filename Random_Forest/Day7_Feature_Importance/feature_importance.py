import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

# Load the wine dataset
wine = load_wine()

# Split the dataset into features and label
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

# Train the Forest Committee
forest = RandomForestClassifier(n_estimators = 100, random_state = 42)
forest.fit(X, y)

# Extract the Feature Importance
importance = forest.feature_importances_

# Create a clean, sorted leaderboard using pandas
leaderboard = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Score": importance,
}).sort_values(by = "Importance_Score", ascending = False).reset_index(drop = True)

print("--- Random Forest Feature Leaderboard ---")
print(leaderboard)