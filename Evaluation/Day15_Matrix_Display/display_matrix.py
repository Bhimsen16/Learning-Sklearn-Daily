import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

dataset = load_breast_cancer()
X = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Generate report 
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names = dataset.target_names))

# Plot the Confusion Matrix
fig, ax = plt.subplots(figsize = (6, 6))
ConfusionMatrixDisplay.from_estimator(
    model, 
    X_test, 
    y_test, 
    display_labels=dataset.target_names, 
    cmap="Blues", 
    ax=ax,
)
plt.title("Confusion Matrix Visualizer")
plt.show()