# Heart Disease Prediction Using Machine Learning
# Project Day 17

# Importing the Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Data Collection & Processing
# -----------------------------

# Load the dataset
data = pd.read_csv('heart_disease_data.csv')

# Check basic info
print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.describe())

# Target distribution
print(data['target'].value_counts())

# -----------------------------
# Splitting Features & Target
# -----------------------------

X = data.drop(columns='target', axis=1)
Y = data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train, Y_train)

# -----------------------------
# Model Evaluation
# -----------------------------

# Training accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data:', training_data_accuracy)

# Testing accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Testing data:', test_data_accuracy)

# -----------------------------
# Building a Predictive System
# -----------------------------

input_data = (45, 1, 3, 110, 264, 0, 1, 132, 0, 1.2, 1, 0, 3)

# Convert input data to DataFrame
input_df = pd.DataFrame(
    [input_data],
    columns=X_train.columns
)

# Prediction
prediction = model.predict(input_df)
print('Prediction:', prediction)

if prediction[0] == 0:
    print('The person does not have Heart Disease')
else:
    print('The person has Heart Disease')
