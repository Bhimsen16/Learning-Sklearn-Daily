import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import the ROC Curve Display
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

dataset = load_breast_cancer()
X = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

# Predict Soft Probabilities instead of hard 0s and 1s
# [:, 1] extracts the probabilities for the positive class (Benign)
y_probabilitites = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_probabilitites)
print("--- Model Performance Benchmark ---")
print(f"Overall ROC-AUC Score: {auc_score * 100:.2f}%\n")

# Set Matplotlib Canvas
fig, ax = plt.subplots(figsize = (7, 5))

# Plot the beautiful ROC Curve visually
RocCurveDisplay.from_estimator(
    model, 
    X_test, 
    y_test, 
    name="Random Forest Engine", 
    ax=ax
)

# Plot a diagonal dashed line representing a useless random guessing model (AUC = 0.5)
plt.plot([0, 1], [0, 1], color="darkgray", linestyle="--", label="Random Guessing (50%)")

# Polish the layout
plt.title("ROC Curve & AUC Tracking")
plt.grid(True)
plt.legend(loc="lower right")
plt.show()