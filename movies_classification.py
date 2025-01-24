import pandas as pd

# Problem Statement:
# The dataset aims to predict whether a movie will win a 'Start Tech Oscar' based on its features.
# The objective is to classify movies into two categories (Oscar or not) based on movie attributes like availability of 3D, Genre, etc.
# This model will help studios or distributors understand which movie features might increase the chances of receiving an Oscar.

# Business Objective:
# The goal is to maximize classification accuracy, assisting movie studios or distributors in understanding 
# which attributes contribute to a movie’s chances of winning a "Start Tech Oscar."

# Load the dataset
df = pd.read_csv(r"E:\Data Science\15-Random Forest\movies_classification.csv")

# Display information about the DataFrame
df.info()

# Comment on DataFrame columns
# The '3D_available' and 'Genre' columns are categorical (object type) and need to be converted into numerical features.
# This can be done using one-hot encoding (dummy variables).

df = pd.get_dummies(df, columns=["3D_available", "Genre"], drop_first=True)

# Assigning input (predictors) and output (target) variables
# The predictors are all columns except 'Start_Tech_Oscar'
predictors = df.loc[:, df.columns != "Start_Tech_Oscar"]
target = df["Start_Tech_Oscar"]

###################################################
from sklearn.model_selection import train_test_split

# Splitting the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2)

###########################################
# Model Selection:
# Using a Random Forest Classifier to make predictions.
# Random Forest is a popular ensemble learning method that aggregates the results of multiple decision trees.

from sklearn.ensemble import RandomForestClassifier

# Creating a RandomForestClassifier model
rand_for = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

# Explanation of hyperparameters:
# n_estimators=500: Increases model accuracy by creating more decision trees. A higher number of trees helps the model generalize better, especially with large datasets.
# n_jobs=-1: Uses all available CPU cores to speed up the training process, crucial when we have a large number of trees.
# random_state=42: Ensures reproducibility of the results, so that the model generates the same output each time it is run.

# Fitting the model on the training dataset
rand_for.fit(X_train, y_train)

# Making predictions on both the training and test datasets
pred_X_train = rand_for.predict(X_train)
pred_X_test = rand_for.predict(X_test)

#########################

# Evaluating the performance of the model on the test dataset

from sklearn.metrics import accuracy_score, confusion_matrix

# Printing accuracy and confusion matrix for the test dataset
print("Test Dataset Accuracy:", accuracy_score(pred_X_test, y_test))
print("Confusion Matrix (Test):\n", confusion_matrix(pred_X_test, y_test))

##########################################

# Evaluating the performance on the training dataset (to check for overfitting or underfitting)

# Printing accuracy and confusion matrix for the training dataset
print("Training Dataset Accuracy:", accuracy_score(pred_X_train, y_train))
print("Confusion Matrix (Training):\n", confusion_matrix(pred_X_train, y_train))

'''
Summary:
- This Random Forest model is used to predict whether a movie will win a 'Start Tech Oscar' based on various features like genre and 3D availability.
- The business goal is to identify which attributes improve a movie's chances of winning an award, providing valuable insights for movie studios and distributors.
- The model's performance is evaluated using both accuracy scores and confusion matrices, both on training and test datasets.
'''

