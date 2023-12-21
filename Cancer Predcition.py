# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the Breast Cancer Wisconsin dataset
# The Breast Cancer dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
# Features include mean radius, mean texture, mean perimeter, etc.
cancer_data = sns.load_dataset('breast_cancer')

# Step 2: Exploratory Data Analysis (EDA)
# Display basic information about the dataset
print("Dataset Information:")
print(cancer_data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(cancer_data.describe())

# Display the first few rows of the dataset
print("\nFirst Few Rows:")
print(cancer_data.head())

# Additional EDA with Seaborn
# Pair plot for pairwise relationships in the dataset
sns.pairplot(cancer_data, hue='diagnosis')
plt.show()

# Box plot to visualize the distribution of features by diagnosis
plt.figure(figsize=(15, 10))
for i, column in enumerate(cancer_data.columns[1:11]):  # Selecting only a subset of features for better visualization
    plt.subplot(2, 5, i + 1)
    sns.boxplot(x='diagnosis', y=column, data=cancer_data)
plt.show()

# Step 3: Data Preprocessing
# Split the data into features (X) and target variable (y)
X = cancer_data.drop('diagnosis', axis=1)
y = cancer_data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train an SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Step 6: Hyperparameter Tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)

# Display the best hyperparameters
print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# Step 7: Predictions on the test set
y_pred = grid_search.predict(X_test_scaled)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Step 9: Prediction on Unseen Data
# Assume a new dataset named 'unseen_data.csv'
unseen_data = pd.read_csv('unseen_data.csv')
unseen_data_scaled = scaler.transform(unseen_data)
unseen_predictions = grid_search.predict(unseen_data_scaled)
print("\nPredictions on Unseen Data:")
print(unseen_predictions)

