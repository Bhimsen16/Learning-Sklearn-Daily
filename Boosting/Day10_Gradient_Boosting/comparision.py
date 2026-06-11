# Comparision between Adaptive and Gradient Boosting

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
x = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
adaptive_model = AdaBoostClassifier(n_estimators = 100, learning_rate = 0.5, random_state = 42)
adaptive_model.fit(x_train, y_train)

gradient_model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, random_state = 42)
gradient_model.fit(x_train, y_train)

adaptive_model_accuracy = accuracy_score(y_test, adaptive_model.predict(x_test))
gradient_model_accuracy = accuracy_score(y_test, gradient_model.predict(x_test))

print("--- Comparision Accuracy Score Between Adaptive and Gradient Boosting ---")
print(f"Adaptive Boosting Model Accuracy: {adaptive_model_accuracy * 100:.2f}%")
print(f"Gradient Boosting Model Accuracy: {gradient_model_accuracy * 100:.2f}%")
