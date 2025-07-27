import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve, auc


#load the dataset

diabetes = load_diabetes()
X,y = diabetes.data, diabetes.target

#convert target variables to binary 0 or 1 (1 for d, 0 for no)
y_binary = (y > np.median(y)).astype(int)

#split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y_binary,test_size=0.2,random_state=42)


#standardize features:
    #normalizes data and determine its mean and standard deviation using the training set
    #then it standardizes the testing data using the mean and sd
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#train the model
model = LogisticRegression()
model.fit(X_train, y_train)

#evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))



