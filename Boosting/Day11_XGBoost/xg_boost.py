import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#Import XGBoost 
from xgboost import XGBClassifier

wine = load_wine()
x = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size = 0.3, random_state = 42)

gradient_model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.8, random_state = 42)
gradient_model.fit(x_train, y_train)

xgradient_model = XGBClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.8, random_state = 42)
xgradient_model.fit(x_train, y_train)

gradient_model_accuracy = accuracy_score(y_test, gradient_model.predict(x_test))
xgradient_model_accuracy = accuracy_score(y_test, xgradient_model.predict(x_test))

print("--- Comparision Between Gradient and Extreme Gradient Boosting ---")
print(f"Gradient Model Accuracy: {gradient_model_accuracy * 100:.2f}%")
print(f"Extreme Gradient Model Accuracy (XGBoost): {xgradient_model_accuracy * 100:.2f}%")