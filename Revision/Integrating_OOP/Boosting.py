# Practice on Boosting Algorithms

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from ModelTrainer import ModelTrainer

# Load data 
wine = load_wine()
X = pd.DataFrame(wine.data, columns= wine.feature_names)
y = wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Instantiate Boosting Models using ModelTrainer
ada_trainer = ModelTrainer(AdaBoostClassifier, n_estimators = 50, random_state = 42)
grb_trainer = ModelTrainer(GradientBoostingClassifier, n_estimators = 50, random_state = 42)

# Train, Evaluate and get Features
ada_trainer.train(X_train, y_train)
ada_trainer.evaluate(X_test, y_test)
ada_trainer.get_feature_importance(wine.feature_names)

grb_trainer.train(X_train, y_train)
grb_trainer.evaluate(X_test, y_test)
grb_trainer.get_feature_importance(wine.feature_names)