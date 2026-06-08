import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load wine dataset
wine = load_wine()
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

# Traditional Train_Test_Split for comparision
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Train a forest with OOB Scoring turned on
forest = RandomForestClassifier(n_estimators = 100, oob_score= True, random_state = 42)
forest.fit(X_train, y_train)

# Extract both scores
traditional_acc = accuracy_score(y_test, forest.predict(X_test))
oob_acc = forest.oob_score_

print("--- Random Forest OOB Test Experiment ---")
print(f"Traditional Test Accuracy: {traditional_acc * 100:.2f}%")
print(f"Built-in OOB Validation Score: {oob_acc * 100:.2f}%")