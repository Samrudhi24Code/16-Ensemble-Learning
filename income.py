# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:08:49 2024

@author: Dell
"""

'''
AdaBoost (Adaptive Boosting) is an ensemble learning 
method designed to improve the performance of weak 
classifiers by combining them to create
a strong classifier.'''
 
 
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
loan_data.columns
loan_data.head()

# Let us split the data into input (X) and output (y)
X = loan_data.iloc[:, 0:6]
y = loan_data.iloc[:, 6]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the AdaBoost classifier with the default base estimator (DecisionTreeClassifier)
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1)

# Train the model
model = ada_model.fit(X_train, y_train)

# Predict the result
y_pred = model.predict(X_test)

# Print accuracy
print("Accuracy with default DecisionTreeClassifier:", metrics.accuracy_score(y_test, y_pred))


# Now, let's try another base model (Logistic Regression)

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



