from sklearn.tree import DecisionTreeClassifier

#Training data to model (Price in $, rating out of 5)
x_train = [
    [4, 4.5],
    [20, 4],
    [110, 4.7],
    [5, 4.6],
    [90, 4.4],
    [6, 3.2],
    [50, 3],
    [120, 2],
]

#1 best seller, 0 not a best seller 
y_train = [1, 1, 1, 1, 1, 0, 0, 0]

#Initialize the ML model
model = DecisionTreeClassifier()

#Train the model
model.fit(x_train, y_train)

print("Model training complete. Now, let's test it's brain...")

#Make a prediction on a raw and unseen data
new_data = [
    [12, 4.4],
    [85, 2.5],
]

predictions = model.predict(new_data)

print(f"Prediction for item 1 (12$, 4.4 rating): {predictions[0]}")
print(f"Prediction for iten=m 2 (85$, 2.5 ratings): {predictions[1]}")