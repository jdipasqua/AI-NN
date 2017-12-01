# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:50:01 2017

@author: jorge
"""

import pandas as pd
#para dividir los datos en train y test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn import svm, datasets
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
tuned_parameters = [{'kernel':('linear', )}]

clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, n_jobs=4, refit=True)

print(clf.fit(X_train, y_train))

