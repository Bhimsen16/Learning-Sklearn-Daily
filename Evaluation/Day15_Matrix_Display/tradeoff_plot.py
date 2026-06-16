import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# New tool: Import Precision-Recall curve display
from sklearn.metrics import PrecisionRecallDisplay

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = Malignant, 1 = Benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(7, 5))
# Plot the Precision-Recall Curve
# This tool automatically calculates precision and recall at different decision thresholds
PrecisionRecallDisplay.from_estimator(
    model, 
    X_test, 
    y_test, 
    name="Random Forest", 
    ax=ax
)

# Titles and labels
plt.title("Day 15: Precision vs. Recall Trade-Off Curve")
plt.grid(True)  # Background grid lines for easy reading
plt.show()      # Renders the window on the screen