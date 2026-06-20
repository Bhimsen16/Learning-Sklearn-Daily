import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Import Cross-Validation score
from sklearn.model_selection import cross_val_score

dataset = load_breast_cancer()
X = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
y = dataset.target

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state = 42)),
])

scores = cross_val_score(pipeline, X, y, cv = 5, scoring = "accuracy")

print("--- 5 Fold Cross-Validation Results ---")
print(f"Score for each fold: {scores}")
print(f"Individual Percentages: {[f'{s*100:.2f}%' for s in scores]}")
print(f"Mean Accuracy: {np.mean(scores) * 100:.2f}%")
print(f"Standard Deviation (Variance): {np.std(scores) * 100:.2f}%")