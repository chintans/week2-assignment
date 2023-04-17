# Exercise 1: Importing Libraries and Loading Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
heart_data = pd.read_csv(url, names=column_names, na_values="?")
print(heart_data.head())

# Exercise 2: Data Exploration
print(heart_data.info())
print(heart_data.describe())
print(heart_data.isna().sum())
sns.pairplot(heart_data, diag_kind="kde")
plt.show()

# Exercise 3: Data Cleaning
heart_data.drop_duplicates(inplace=True)
heart_data.fillna(heart_data.median(), inplace=True)

# Exercise 4: Feature Scaling and Encoding
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
heart_data[numerical_columns] = scaler.fit_transform(heart_data[numerical_columns])

# Exercise 5: Dataset Splitting
from sklearn.model_selection import train_test_split

X = heart_data.drop("target", axis=1)
y = heart_data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
