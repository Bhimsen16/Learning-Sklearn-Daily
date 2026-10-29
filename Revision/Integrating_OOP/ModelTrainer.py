# Practice on building OOP model trainer

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, model_class, **model_param):
        # Self is used to save model on the instance so it remains even after __init__ ends
        self.model = model_class(**model_param)
        self.is_trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"\n{self.model.__class__.__name__} trained successfully!!")

    def evaluate(self, X_test, y_test):
        if self.is_trained:
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f"{self.model.__class__.__name__}'s Accuracy = {accuracy * 100:.2f}%")
            return accuracy

        else:
            print(f"{self.model.__class__.__name__} is not trained. Train the model first!!")
            return None

    # Extract top features
    def get_feature_importance(self, feature_names):
        if self.is_trained:
            importances = pd.Series(self.model.feature_importances_, index= feature_names)

            # Filter features that have non-zero importance and sort it descending
            top_features = importances[importances > 0].sort_values(ascending= False)
            print(f"\n--- {self.model.__class__.__name__} Top Features ---")
            print(top_features)
            return top_features

        else:
            print("Train the model first!!")
            return None

    # Plot Decision Flowchart
    def visualize_tree(self, feature_names, class_names):
        if self.is_trained and isinstance(self.model, DecisionTreeClassifier):
            plt.figure(figsize= (14, 8))
            plot_tree(
                self.model, 
                feature_names= feature_names, 
                class_names= class_names,
                filled= True, # Color nodes by majority class
                rounded= True, # Roudned boxes
                fontsize= 12,
            )
            plt.title("Decision Tree Decision Flowchart")
            plt.show() 
        elif not isinstance(self.model, DecisionTreeClassifier):
            print("Tree visualization is specifically tailored for single DecisionTreeClassifier model!")

        else:
            print("Train the model first!")

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns= wine.feature_names)
y = wine.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Instantiate ModelTrainer with Decision Tree
trainer = ModelTrainer(DecisionTreeClassifier, max_depth = 3, random_state = 42)

# Instantiate ModelTrainer with Random Forest
trainer2 = ModelTrainer(RandomForestClassifier, max_depth = 3, random_state = 42)

# Evaluating before training
trainer.evaluate(X_test, y_test)

# Train and Evaluate
trainer.train(X_train, y_train)
trainer.evaluate(X_test, y_test)
trainer.get_feature_importance(wine.feature_names)
trainer.visualize_tree(feature_names= wine.feature_names, class_names= wine.target_names)

trainer2.train(X_train, y_train)
trainer2.evaluate(X_test, y_test)
trainer2.get_feature_importance(wine.feature_names)
trainer2.visualize_tree(feature_names= wine.feature_names, class_names= wine.target_names)