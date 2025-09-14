import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # These are our features (sepal length, petal width, etc.)
y = iris.target # These are our labels (the species, 0, 1, or 2)

print("Shape of features (X):", X.shape)
print("Shape of labels (y):", y.shape)

# 2. Split the data into training and testing sets
# We typically use 70-80% for training and 20-30% for testing.
# random_state ensures we get the same split every time for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nShape of training features (X_train):", X_train.shape)
print("Shape of testing features (X_test):", X_test.shape)
print("Shape of training labels (y_train):", y_train.shape)
print("Shape of testing labels (y_test):", y_test.shape)

# 3. Create (Instantiate) the K-Nearest Neighbors Classifier model
# We'll start with k=3 neighbors.
knn = KNeighborsClassifier(n_neighbors=3)

# 4. Train the model (the "learning" step!)
# The model learns the patterns from the training features (X_train)
# and their corresponding labels (y_train).
knn.fit(X_train, y_train)

print("\nModel training complete!")

# 5. Make predictions on the unseen test data
# The model now uses what it learned to predict labels for X_test
y_pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

# Calculate and print the confusion matrix
# This matrix gives you the raw counts of TP, TN, FP, FN
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Generate a classification report
# This provides Precision, Recall, F1-Score, and Support for each class
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(report)

print("\nFirst 5 true labels (y_test):   ", y_test[:5])
print("First 5 predicted labels (y_pred):", y_pred[:5])
print("First 5 predicted lables (y_pred):",iris.target_names[y_pred[:5]])

# 6. Evaluate the model's performance
# We compare the model's predictions (y_pred) with the actual labels (y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the KNN model: {accuracy:.2f}")
