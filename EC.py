#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier


# In[3]:


df = pd.read_csv("heart.csv")
print(df.describe())

print("-------------------------------------------------------------")

# Splitting the data - training and test data (80-20)

# print(df.dtypes)




X = df.drop(columns = ['output'])
Y = df['output']

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, stratify = Y)

print("-------------------------------------------------------------")

# Naive Bayes Classification

nbModel = GaussianNB()
nbModel.fit(xTrain, yTrain)
nbPred = nbModel.predict(xTest)

print("Naive Bayes Accuracy:", nbModel.score(xTest, yTest))
print("Confusion Matrix:")
print("", confusion_matrix(yTest, nbPred))
nbAccuracy=np.average(cross_val_score(nbModel, X, Y, cv = 10))
print("Average 10-Fold Validation Score:", nbAccuracy)

print("-------------------------------------------------------------")


# Logistic Regression


lrModel = LogisticRegression()
lrModel.fit(xTrain, yTrain)
lrPred = lrModel.predict(xTest)

print("Logistic Regression Accuracy:", lrModel.score(xTest, yTest))
print("Confusion Matrix:")
print("", confusion_matrix(yTest, lrPred))
lrAccuracy=np.average(cross_val_score(lrModel, X, Y, cv = 10))
print("Average 10-Fold Validation Score:", lrAccuracy)

print("-------------------------------------------------------------")

# Decision Tree Classifier


dtModel = DecisionTreeClassifier()
dtModel.fit(xTrain, yTrain)
dtPred = dtModel.predict(xTest)

print("Decision Tree Accuracy:", dtModel.score(xTest, yTest))
print("Confusion Matrix:")
print("", confusion_matrix(yTest, dtPred))
dtAccuracy=np.average(cross_val_score(dtModel, X, Y, cv = 10))
print("Average 10-Fold Validation Score:", dtAccuracy)
plot_tree(dtModel)

print("-------------------------------------------------------------")
#Ensemble Model


estimators = [('Logistic Regression', lrModel), ('Naive Bayes', nbModel), ('Decision Tree', dtModel)]

eModel = VotingClassifier(estimators, voting = 'hard')
eModel.fit(xTrain, yTrain)
ePred = eModel.predict(xTest)

print("Ensemble Model Accuracy:", eModel.score(xTest, yTest))
print("Confusion Matrix:")
print("", confusion_matrix(yTest, ePred))
print("10-Fold Validation Scores:\n", cross_val_score(eModel, X, Y, cv = 10))
print("Average 10-Fold Validation Score (Hard Voting):", np.average(cross_val_score(eModel, X, Y, cv = 10)))

print("-------------------------------------------------------------")

sModel = VotingClassifier(estimators, voting = 'soft',weights=[1,0.75,0.25])
sModel.fit(xTrain, yTrain)
sPred = sModel.predict(xTest)

print("Ensemble Model Accuracy:",sModel.score(xTest, yTest))
print("Confusion Matrix:")
print("", confusion_matrix(yTest, sPred))
print("10-Fold Validation Scores:\n", cross_val_score(sModel, X, Y, cv = 10))
print("Average 10-Fold Validation Score (Soft Voting):", np.average(cross_val_score(sModel, X, Y, cv = 10)))


print("-------------------------------------------------------------")


