#Import necessary tools 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Load the iris dataset directly 
iris = load_iris()

#Convert the dataset into Pandas dataframe 
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df["species"] = iris.target #The target is the flower species (0, 1, or 2)

print("--- Real Dataset Loaded ---")
print(f"Dataset Shape: {df.shape} (150 Flowers, 5 Columns!)")
print("\nThe first 5 rows of data:")
print(df.head())
print("-" * 50)

#Split into features(X) and label(y)
X = df[iris.feature_names]
y = df["species"]

#The split (25% for the final test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Train the restricted tree
model = DecisionTreeClassifier(max_depth = 2, random_state = 42)
model.fit(X_train, y_train)

#Evaluate on the test set
test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

print("--- Evaluation Results ---")
print(f"Model Accuracy Score on Real Data: {accuracy * 100:.1f}%")

#Save the real-world tree visualization
plt.figure(figsize = (10, 6))
plot_tree(
    model,
    feature_names = iris.feature_names,
    class_names = iris.target_names,
    filled = True,
    rounded = True,
)

plt.title("Real-World Iris Flower Classifier")
plt.savefig("iris_tree.png", bbox_inches = "tight")
print("\nReal-World tree visualization saved as 'iris_tree.png'!")