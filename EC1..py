import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

data1 = pd.read_csv('data_1.csv')
data2 = pd.read_csv('data_2.csv')
df = pd.concat([data1, data2])


# df = df.loc[:, (df != 0).any(axis=0)]
print(len(df.columns))
df = df.dropna()
print(len(df))
X = df.drop(['Label','Circuit'], axis=1)
Y = df['Label']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,stratify=Y)


lr = LogisticRegression()
lr.fit(x_train,y_train)
ypred = lr.predict(x_test)
lr.score(x_test,y_test)
print(classification_report(ypred,y_test))

dt = DecisionTreeClassifier(criterion="gini")
dt.fit(x_train,y_train)
ypred = dt.predict(x_test)
dt.score(x_test,y_test)
print(classification_report(ypred,y_test))


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
tree.plot_tree(dt,feature_names=list(X.columns),class_names=np.unique(Y).astype(str),filled=True)
plt.show()
accuracy_score(y_test,ypred)


nb = GaussianNB()
nb.fit(x_train,y_train)
ypred = nb.predict(x_test)
print(nb.score(x_test,y_test))
print(classification_report(ypred,y_test))


models = [('Logistic Regression',lr),('Decision Tree',dt),('Naive Bayes',nb)]
ensemble = VotingClassifier(estimators=models,voting='hard')
ensemble.fit(x_train,y_train)
ypred = ensemble.predict(x_test)
print(ensemble.score(x_test,y_test))
print(ensemble.score(x_test,y_test))
print(classification_report(ypred,y_test))

models = [('Logistic Regression',lr),('Decision Tree',dt),('Naive Bayes',nb)]
ensemble = VotingClassifier(estimators=models,voting='soft',weights=[1,0.75,0.25])
ensemble.fit(x_train,y_train)
ypred = ensemble.predict(x_test)
print(ensemble.score(x_test,y_test))
print(ensemble.score(x_test,y_test))
print(classification_report(ypred,y_test))

models = [('Logistic Regression',lr),('Decision Tree',dt),('Naive Bayes',nb)]
ensemble = RandomForestClassifier(n_estimators=40)
ensemble.fit(x_train,y_train)
ypred = ensemble.predict(x_test)
print(ensemble.score(x_test,y_test))
print(ensemble.score(x_test,y_test))
print(classification_report(ypred,y_test))

models = [('Logistic Regression',lr),('Decision Tree',dt),('Naive Bayes',nb)]
ensemble = AdaBoostClassifier(n_estimators=40)
ensemble.fit(x_train,y_train)
ypred = ensemble.predict(x_test)
print(ensemble.score(x_test,y_test))
print(ensemble.score(x_test,y_test))
print(classification_report(ypred,y_test))

models = [('Logistic Regression',lr),('Decision Tree',dt),('Naive Bayes',nb)]
ensemble = XGBClassifier(n_estimators=40)
ensemble.fit(x_train,LabelEncoder().fit_transform(y_train))
ypred = ensemble.predict(x_test)
print(ensemble.score(x_test,LabelEncoder().fit_transform(y_test)))
print(classification_report(ypred,LabelEncoder().fit_transform(y_test)))

  
