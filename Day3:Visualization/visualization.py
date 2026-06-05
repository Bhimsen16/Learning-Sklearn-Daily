#Import necessary tools or libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

#Custom dataset 
data = {
    "prices": [12, 34, 44, 10, 5, 7, 100, 67, 43, 55, 90, 44, 65, 76],
    "ratings": [1.1, 4.0, 4.7, 4.7, 4.4, 2.0, 4.1, 3.1, 4.0, 4.8, 2.2, 3.4, 3.0, 4.1],
    "best_seller": [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
}

df = pd.DataFrame(data)

#Split the dataset into features and labels
X = df[["prices", "ratings"]]
y = df["best_seller"]

#The SPLIT(Added random_state=42 to keep splits identical)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Train the model
model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

#Visualing the decison tree using Matplotlib
plt.figure(figsize = (10, 6))
plot_tree(
    model, 
    feature_names = ["prices", "ratings"],
    class_names = ["Not Best Seller", "Best Seller"],
    filled = True,
    rounded = True,
)
plt.title("Inside the Decision Tree's Brain")
plt.savefig("decision_tree_day_24.png", bbox_inches = "tight")
print("Decision Tree Visualization saved as 'decision_tree_day-24.png'!")

new_products = pd.DataFrame({
    "prices": [8, 85],
    "ratings": ["4.9", "2.5"],
})

print("\n------Testing on Brand New Products------\n")
print(new_products)

#Ask the model to predict their successes
new_predictions = model.predict(new_products)

print("\n------AI predictions------")
for i, pred in enumerate(new_predictions):
    status = "Best Seller! " if pred == 1 else "Not a Best Seller. "
    print(f"Product {i+1} (Price: ${new_products['prices'].iloc[i]}, Rating: {new_products['ratings'].iloc[i]}): Prediction -> {status}")