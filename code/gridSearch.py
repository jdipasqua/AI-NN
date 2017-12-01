# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:50:01 2017

@author: jorge
"""

import pandas as pd
#para dividir los datos en train y test
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report,confusion_matrix



train = pd.read_csv('../data/fashion-mnist_train.csv')

train.head()

train.describe().transpose()

X = train.drop('label',axis=1)

y = train['label']


#divide los datos de entrenamiento y los datos de pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y)


################################################################################
# Set the parameters by cross-validation
param_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(svm.SVC(C=1), param_grid, n_jobs=4, refit=True)

clf.fit(X_train, y_train)

print("------fin-----")

