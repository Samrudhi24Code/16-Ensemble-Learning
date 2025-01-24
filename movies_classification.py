import pandas as pd

# Correct file path
df = pd.read_csv(r"E:\Data Science\15-Random Forest\movies_classification.csv")

# Display information about the DataFrame
df.info()
#movies classification dataset contain two columns which are object
#hence convert into dummies

df=pd.get_dummies(df,columns=["3D_available","Genre"],drop_first="True")
#let us assign input and output variable
# Selecting the 'Start_Tech_Oscar' column from the DataFrame
predictors = df.loc[:, df.columns!="Start_Tech_Oscar"]
target=df["Start_Tech_Oscar"]
###################################################
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(predictors, target,test_size=0.2)
###########################################
#model selection

from sklearn.ensemble import RandomForestClassifier
rand_for=RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)

'''n_estimators=500: Increases model accuracy by building more trees, especially if your dataset is large or complex.
n_jobs=-1: Utilizes all available CPU cores to train the model faster, useful for big datasets or when you have many trees (n_estimators is large).
random_state=42: Ensures that you get the same results every time, which is crucial when tuning hyperparameters or comparing model performance.'''
rand_for.fit(X_train,y_train)
pred_X_train=rand_for.predict(X_train)
pred_X_test=rand_for.predict(X_test)
#########################

#let us check the performance of the model
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(pred_X_test,y_test)
confusion_matrix(pred_X_test,y_test)

##########################################
#for training dataset

accuracy_score(pred_X_train,y_train)
confusion_matrix(pred_X_train,y_train)
###############################