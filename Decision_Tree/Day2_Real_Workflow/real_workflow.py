#1. Import necessary tools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#2. Create a larger dataset using Pandas 
data = {
    "prices": [12, 34, 44, 10, 5, 7, 100, 67, 43, 55, 90, 44, 65, 76],
    "ratings": [1.1, 4.0, 4.7, 4.7, 4.4, 2.0, 4.1, 3.1, 4.0, 4.8, 2.2, 3.4, 3.0, 4.1],
    "best_seller": [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
}

df = pd.DataFrame(data)
print("Full Dataset:")
print(df)
print("-"* 40)

#3. Split into features(X) and label(y)
X = df[["prices", "ratings"]]
y = df["best_seller"]

#4. THE SPLIT: Hide 25% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)

print("Data Split Complete.")
print(f"Training shapes: X_train= {X_train.shape}, y_train= {y_train.shape}")
print(f"Testing shapes: X_test= {X_test.shape}, y_test= {y_test.shape}\n")

# 5. Train the model on the Training Set only
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 6. Evaluate: Ask the model to predict the hidden test set
predictions = model.predict(X_test)

# 7. Calculate accuracy score
# Compares what the model guessed (predictions) vs the actual true answers (y_test)
accuracy = accuracy_score(y_test, predictions)

print("Evaluation results:")
print(f"Actual answers (y_test): {y_test.to_list()}")
print(f"Model's guesses (predicted): {list(predictions)}")
print(f"Accuracy score: {accuracy * 100 :.1f}%")