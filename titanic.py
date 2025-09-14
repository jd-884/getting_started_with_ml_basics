import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load the Titanic dataset ---
# Make sure 'train.csv' is in the same directory as your script,
# or provide the correct path (e.g., 'path/to/titanic/train.csv')
try:
    data = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the file is in the correct directory.")
    print("If it's in a subfolder (e.g., 'titanic/train.csv'), adjust the path in pd.read_csv().")
    exit() # Exit the script if file is not found

# --- 2. Data Cleaning and Preprocessing ---

# Drop columns that are not directly useful for a basic model or have too many missing values
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
print("\nDropped 'PassengerId', 'Name', 'Ticket', 'Cabin' columns.")
print("DataFrame shape after dropping columns:", data.shape)

# Drop rows with missing values in 'Age' or 'Embarked'
# (Age has some missing, Embarked has very few)
data.dropna(subset=['Age', 'Embarked'], inplace=True)
print("DataFrame shape after dropping rows with missing Age/Embarked:", data.shape)

# Convert 'Sex' to numerical (male=1, female=0)
sex_mapping = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_mapping)
print("\n'Sex' column converted to numerical (male=1, female=0).")
print(data[['Sex']].head())

# One-Hot Encoding for 'Embarked'
# Creates new columns like 'Embarked_Q' and 'Embarked_S'
# drop_first=True avoids multicollinearity by dropping one of the dummy variables (e.g., Embarked_C)
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
print("\n'Embarked' column One-Hot Encoded.")
print("First 5 rows of data after One-Hot Encoding 'Embarked':")
print(data.head())

# Display final cleaned data info
print("\nFinal info after all preprocessing steps:")
data.info()

# --- 3. Define Features (X) and Labels (y) ---
# X contains all columns except 'Survived'
X = data.drop('Survived', axis=1)
# y contains only the 'Survived' column
y = data['Survived']

print(f"\nShape of X (features): {X.shape}")
print(f"Shape of y (labels): {y.shape}")

# --- 4. Split Data into Training and Testing Sets ---
# 80% for training, 20% for testing
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- 5. Feature Scaling ---
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler ONLY on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully using StandardScaler.")
print("First 5 rows of X_train_scaled (note the new values):")
print(X_train_scaled[:5]) # This will be a NumPy array, not a DataFrame

# --- 6. Initial Model Training (without tuning, just for comparison) ---
knn_initial = KNeighborsClassifier(n_neighbors=5)
knn_initial.fit(X_train_scaled, y_train) # Use scaled data for training
print("\nInitial KNeighborsClassifier model trained successfully with scaled data.")

# Evaluate initial model
y_pred_initial = knn_initial.predict(X_test_scaled)
accuracy_initial = accuracy_score(y_test, y_pred_initial)
print(f"\nInitial Model Accuracy (n_neighbors=5, scaled data): {accuracy_initial:.4f}")
print("\nInitial Classification Report (Scaled Model):")
print(classification_report(y_test, y_pred_initial, target_names=['Did Not Survive', 'Survived']))


# --- 7. Hyperparameter Tuning with GridSearchCV ---
print("\n--- Starting Hyperparameter Tuning with GridSearchCV ---")

# Create a fresh KNeighborsClassifier instance for GridSearchCV
knn_grid_search_estimator = KNeighborsClassifier()

# Define the parameter grid to search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19], # Testing a range of K values
    'weights': ['uniform', 'distance'],              # How to weight neighbors
    'metric': ['euclidean', 'manhattan']             # Distance calculation method
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=knn_grid_search_estimator,
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    scoring='accuracy', # Metric to optimize
    verbose=1,          # Show progress
    n_jobs=-1           # Use all available CPU cores
)

# Run the Grid Search on the SCALED TRAINING DATA
# This step will train many models internally to find the best parameters
grid_search.fit(X_train_scaled, y_train)

print("\nGrid Search completed!")

# --- Get the Best Parameters and Best Score ---
print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")
print(f"Best cross-validation accuracy on training data: {grid_search.best_score_:.4f}")

# --- Evaluate the Best Model on the Test Set ---
# grid_search.best_estimator_ automatically holds the model trained with the best parameters
best_knn_model = grid_search.best_estimator_
y_pred_tuned = best_knn_model.predict(X_test_scaled)

print(f"\nTest set accuracy with BEST (tuned) parameters: {accuracy_score(y_test, y_pred_tuned):.4f}")
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned, target_names=['Did Not Survive', 'Survived']))

print("\n--- End of Machine Learning Project Workflow ---")