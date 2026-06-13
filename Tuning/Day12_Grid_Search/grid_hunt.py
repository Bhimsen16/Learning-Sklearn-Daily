import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

wine = load_wine()
x = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Initialize the model
xgb_model = XGBClassifier(random_state = 42)

# Define the combination grid to test
test_parameters = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [1, 2, 3],
}

# Set up Grid Search with 5-Fold Cross Validation
# cv=5 means it will test each combination 5 times on different data chunks
grid_search = GridSearchCV(estimator = xgb_model, param_grid = test_parameters, cv = 5, scoring = "accuracy", n_jobs = -1)

print("Hunting for the best hyperparameters....")
grid_search.fit(x_train, y_train)

print("--- Hunt Complete ---\n")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_ * 100:.2f}%")

# Evaluate on the true test set using the best found model configuration
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(x_test, y_test)
print(f"Final Accuracy on Test Set: {test_accuracy * 100:.2f}%")