import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset and split it
wine = load_wine()
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

# Initialize the algorithm
forest = RandomForestClassifier(n_estimators = 100, oob_score = True, random_state = 42)

# Train the model
forest.fit(X, y)

# Calculate the accuracy score
oob_accuracy = forest.oob_score_

# Print the result
print(f"The model got {oob_accuracy * 100:.2f}% accuracy while using OOB Score.")