# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:08:49 2024

@author: Dell
"""

'''
AdaBoost (Adaptive Boosting) is an ensemble learning 
method designed to improve the performance of weak 
classifiers by combining them to create a strong classifier.

# Problem Statement:
To build a predictive model to classify income groups (e.g., high vs. low income) 
based on features like age, education, work experience, etc., using ensemble learning techniques.

# Business Objective:
The objective is to improve classification accuracy, which will help organizations 
make better decisions related to customer segmentation, credit scoring, or targeted marketing.

# Goals:
- **Maximize:** Classification accuracy by combining weak classifiers.
- **Minimize:** Errors in misclassifications.

# Solution:
- Use the AdaBoost algorithm, an ensemble technique that combines multiple weak classifiers 
  (default: Decision Tree) or other base models (e.g., Logistic Regression) to create a 
  strong classifier and evaluate its performance.
'''

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import warnings
import sklearn

print(sklearn.__version__)

warnings.filterwarnings('ignore')

# Load the dataset
loan_data = pd.read_csv(r"E:\Data Science\16-Ensemble Techique\income.csv")
loan_data.columns  # Display the columns for reference
loan_data.head()   # Display the first few rows of the dataset

# Split the dataset into input (X) and output (y)
X = loan_data.iloc[:, 0:6]  # Input features: First 6 columns
y = loan_data.iloc[:, 6]    # Output/target variable: 7th column (income group)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 1: Train AdaBoost with the default base estimator (DecisionTreeClassifier)
# DecisionTreeClassifier is used as the base weak learner for AdaBoost.

# Create the AdaBoost classifier
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1)

# Train the model
model = ada_model.fit(X_train, y_train)

# Predict the result
y_pred = model.predict(X_test)

# Print accuracy
print("Accuracy with default DecisionTreeClassifier:", metrics.accuracy_score(y_test, y_pred))

# Step 2: Train AdaBoost with Logistic Regression as the base learner
# Logistic Regression is used as the base weak learner to test the model's performance.

# Define the base learner (Logistic Regression)
lr = LogisticRegression()

# Create the AdaBoost model with Logistic Regression as the base learner
Ada_model = AdaBoostClassifier(estimator=lr, n_estimators=50, learning_rate=1)

# Train the AdaBoost model with Logistic Regression as the base learner
model = Ada_model.fit(X_train, y_train)

# Predict the result
y_pred = model.predict(X_test)

# Print accuracy
print("Accuracy with Logistic Regression base model:", metrics.accuracy_score(y_test, y_pred))

'''
# Summary:
- The default base model (DecisionTreeClassifier) provides one accuracy score.
- Using Logistic Regression as the base learner in AdaBoost provides another accuracy score.
- These results can be used to determine the better-performing model and help meet the 
  business objective of accurate income classification.
'''
