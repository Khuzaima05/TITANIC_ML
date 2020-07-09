# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:22:58 2020

@author: WiN 10
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test1 = pd.read_csv('test.csv')

def add_age(cols):
 Age = cols[0]
 Pclass = cols[1]
 if pd.isnull(Age):
  return int(train[train["Pclass"] == Pclass]["Age"].mean())
 else:
  return Age

train["Age"] = train[["Age", "Pclass"]].apply(add_age,axis=1)


train.drop("Cabin", inplace=True, axis=1)

train.dropna(inplace= True)

pd.get_dummies(train['Sex'])
sex= pd.get_dummies(train['Sex'],drop_first= True)

pd.get_dummies(train['Embarked'])
embarked = pd.get_dummies(train["Embarked"],drop_first=True)
pd.get_dummies(train['Pclass'])
pclass = pd.get_dummies(train["Pclass"],drop_first=True)

train = pd.concat([train,pclass,sex,embarked],axis=1)

train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)

X=train.drop("Survived",axis =1)
y=train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
 
k= test
predictions = logmodel.predict(k)

ids = test1['PassengerId']

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })




