# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:52 2017

@author: jorge
"""

import pandas as pd
#para dividir los datos en train y test
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix


train = pd.read_csv('../data/fashion-mnist_train.csv')
test = pd.read_csv('../data/fashion-mnist_test.csv')

train.head()

train.describe().transpose()

X = train.drop('label',axis=1)

y = train['label']


#divide los datos de entrenamiento y los datos de pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y)

#clf = ExtraTreesClassifier(n_estimators=20, max_depth=None, min_samples_split=4, random_state=0)

clf = ExtraTreesClassifier(n_estimators=20, max_depth=1000, min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X_train, y_train)

clf.fit(X_train,y_train)

predictions = clf.predict(X_test)


#print(confusion_matrix(y_test,predictions))

#print('\n')

print(classification_report(y_test,predictions))
 

#print(scores.mean())

