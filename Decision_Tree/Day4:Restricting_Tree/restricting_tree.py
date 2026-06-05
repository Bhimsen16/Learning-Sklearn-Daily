import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#The dataset
data = {
    "Prices": [10, 24, 33, 12, 60, 44, 34, 64, 90, 34, 43, 44, 98],
    "Ratings": [2.3, 4.0, 3.0, 4.1, 4.0, 2.1, 2.3, 3.7, 4.1, 4.5, 3.8, 1.0, 0.0],
    "Best_seller": [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
#Split into features(X) and label(y)
X = df[["Prices", "Ratings"]]
y = df["Best_seller"]

#The Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 42)

#Max_depth means the model can ask only given amount max before making final decision
restricted_model = DecisionTreeClassifier(max_depth = 2, random_state = 42)

#Ask the model to learn from this data
restricted_model.fit(X_train, y_train)

#Predict on the new dataset
new_products = pd.DataFrame({
    "Prices": [8, 85],
    "Ratings": [3.8, 3.1],
})

predictions = restricted_model.predict(new_products)

print("-------Restricted Tree (max_depth = 2) Predictions-------")
for i, pred in enumerate(predictions):
    status = "Best Seller" if pred == 1 else "Not a Best Seller"
    print(f"Product {i + 1} (Price: ${new_products["Prices"].iloc[i]}, Rating: {new_products["Ratings"].iloc[i]}) : Prediction -> {status}")

#Save the new restricted image
plt.figure(figsize= (8, 5))
plot_tree(
    restricted_model,
    feature_names = ["Prices", "Ratings"],
    class_names = ["Not a Best Seller", "Best Seller"],
    filled = True, 
    rounded = True,
)

plt.title("Restricted Tree (max_depth = 2)")
plt.savefig("restricted_tree.png",bbox_inches = "tight")
print("\nNew tree visualization saved as 'restricted_tree.png'!")

# --- THE FIX HAPPENS HERE ---

# 2. Predict on the actual TEST SET to see how well it learned
test_predictions = restricted_model.predict(X_test)

# 3. Calculate accuracy score by comparing y_test (4 items) vs test_predictions (4 items)
accuracy = accuracy_score(y_test, test_predictions)
print(f"\nModel Evaluation Accuracy Score: {accuracy * 100 :.1f}%")