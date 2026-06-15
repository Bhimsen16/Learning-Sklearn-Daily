import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import for evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Load data (Binary classification: Malignant or Benign)
dataset = load_breast_cancer()
x = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Generate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("--- Raw Confusion Matrix ---")
print(cm)

# Generate the complete classification report (Precision, Recall, F1-Score)
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=dataset.target_names))