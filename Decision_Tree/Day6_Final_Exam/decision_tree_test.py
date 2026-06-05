#Import necessary tools
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Load the wine dataset
wine = load_wine()

#Split the dataset into features (x) and label (y)
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
y = wine.target

# Task 1: Split the dataset keeping 30% data for testing with random state 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state= 42)

# Task 2: Create a restricted Decision Tree with depth 3
model = DecisionTreeClassifier(max_depth = 3, random_state = 42)

# Task 3: Train the model on training data
model.fit(X_train, y_train)

# Task 4: Evaluate the model
test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

print("--- Evaluation Results ---")
print(f"Accuracy Score of the Model: {accuracy * 100:.1f}%")

# Task 5: Save the visual chart of the tree
plt.figure(figsize = (10, 6))
plot_tree(
    model,
    feature_names = wine.feature_names,
    class_names = wine.target_names,
    filled = True, 
    rounded = True,
)

plt.title("Wine Classifier")
plt.savefig("wine_tree.png", bbox_inches = "tight")
print("Test Complete! Tree diagram saved.")