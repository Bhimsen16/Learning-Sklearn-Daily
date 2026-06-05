import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Load the dataset
iris = load_iris()

#Split it into features and label
X = pd.DataFrame(data = iris.data, columns = iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)

#Train the model
model = DecisionTreeClassifier(max_depth= 2, random_state = 42)
model.fit(X_train, y_train)

#EXTRACT MODEL IMPORTANCES
#This returns an array of scores matching the order of iris.feature_names
importances = model.feature_importances_

print("--- Raw Importance Score ---")
print(importances)
print("-" * 30)

#Make it look beautiful using Pandas
importance_df = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance Score (%)": importances * 100
})

#Sort them so the most important feature is at top
importance_df = importance_df.sort_values(by="Importance Score (%)", ascending= False)

print("\n---Decision Tree's Priority List ---")
print(importance_df.to_string(index=False))