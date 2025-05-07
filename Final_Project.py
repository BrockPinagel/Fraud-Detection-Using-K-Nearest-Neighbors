# Brock Pinagel
# 12/16/2024
# Final Project
# INT 7623 Data Science for Business

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'bank_transactions_data_2.csv'  
data = pd.read_csv(file_path)

# # Check data summary
print(data.info())

# Graphs
# Visualize correlations among numerical features
numerical_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Scatter plot: TransactionAmount vs. AccountBalance
sns.scatterplot(data=data, x='TransactionAmount', y='AccountBalance', hue='TransactionType', alpha=0.7)
plt.title("Transaction Amount vs. Account Balance")
plt.xlabel("Transaction Amount")
plt.ylabel("Account Balance")
plt.legend(title="Transaction Type")
plt.show()

# Countplot for TransactionType
sns.countplot(x="TransactionType", data=data, palette="pastel", hue="TransactionType")
plt.title("Distribution of Transaction Types")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

# Preprocessing
class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to datetime
        X_parsed = pd.to_datetime(X.squeeze(), errors='coerce')  # Ensure single-column input
        return pd.DataFrame({
            'hour': X_parsed.dt.hour.fillna(-1),  # Extract hour; fill NaN with -1
            'day_of_week': X_parsed.dt.dayofweek.fillna(-1)  # Extract day of the week; fill NaN with -1
        })

# Define columns
numerical_columns = ['TransactionAmount', 'AccountBalance', 'TransactionDuration']
categorical_columns = ['TransactionType', 'Channel', 'Location', 'CustomerOccupation']
date_columns = ['TransactionDate']

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

date_transformer = Pipeline(steps=[
    ('date_features', DateTransformer()),
    ('scaler', StandardScaler())
])

# Combine preprocessors
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns),
    ('date', date_transformer, date_columns)
])

# Separate features and target
X = data.drop(columns=['TransactionID', 'PreviousTransactionDate'])
y = data['TransactionType']  

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Data Partitioning
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

# k Values
k_values = [3, 5, 7]
print(f"Chosen K values: {k_values}")


# Train K-NN
models = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    models[k] = knn

# Test Model
test_accuracies = {}
for k, model in models.items():
    y_pred = model.predict(X_test)
    test_accuracies[k] = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K={k}: {test_accuracies[k]:.2f}")

# Evaluation
for k, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Evaluation for K={k}:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")

# Best Model
best_k = max(test_accuracies, key=test_accuracies.get)
print(f"The best model is with K={best_k} and accuracy={test_accuracies[best_k]:.2f}")
