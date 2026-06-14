import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

wine = load_wine()
x = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

xgb_model = XGBClassifier(random_state = 42)

# Define a wider parameter grid
param_grid = {
    "n_estimators": [50, 100, 150, 200, 250],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    "max_depth": [1, 2, 3, 4, 5],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
}

# n_iter=10 means it will randomly pick exactly 10 combinations to try, out of 750 total possibilities!
random_search = RandomizedSearchCV(
    estimator = xgb_model,
    param_distributions = param_grid,
    n_iter = 10,
    cv = 5, 
    scoring = "accuracy",
    random_state = 42,
    n_jobs = -1,
)

print("Running a calculated random hunt over 750 possibilitites.")
random_search.fit(x_train, y_train)

print("\n--- Random Hunt Complete ---\n")
print(f"Best Parameters Found: {random_search.best_params_}")
print(f"Best Cross-Validation Score: {random_search.best_score_ * 100:.2f}%")

# Evaluate on test set
best_model = random_search.best_estimator_
print(f"Final Accuracy on Test Set: {best_model.score(x_test, y_test) * 100:.2f}%")