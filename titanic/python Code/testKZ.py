# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:09:01 2020

@author: WiN 10
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv('test.csv')


def add_age(cols):
 Age = cols[0]
 Pclass = cols[1]
 if pd.isnull(Age):
  return int(test[test["Pclass"] == Pclass]["Age"].mean())
 else:
  return Age

test["Age"] = test[["Age", "Pclass"]].apply(add_age,axis=1)


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,4])
X[:, 1:3] = imputer.transform(test[:,4])

test.drop("Cabin", inplace=True, axis=1)


pd.get_dummies(test['Sex'])
sex= pd.get_dummies(test['Sex'],drop_first= True)

pd.get_dummies(test['Embarked'])
embarked = pd.get_dummies(test["Embarked"],drop_first=True)
pd.get_dummies(test['Pclass'])
pclass = pd.get_dummies(test["Pclass"],drop_first=True)

test = pd.concat([test,pclass,sex,embarked],axis=1)

test.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)

#set ids as PassengerId and predict survival 


#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

