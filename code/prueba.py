import pandas as pd
#para dividir los datos en train y test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix



train = pd.read_csv('../data/fashion-mnist_train.csv')
test = pd.read_csv('../data/fashion-mnist_test.csv')

train.head()

train.describe().transpose()

X = train.drop('label',axis=1)

y = train['label']


#divide los datos de entrenamiento y los datos de pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y)

#print(X_train)

#normalizacion de datos
#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

# Fit only to the training data
#scaler.fit(X_train)





# Now apply the transformations to the data:
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

#modelo de mi clasificador
mlp = MLPClassifier(hidden_layer_sizes=(24,24,24,24,24),max_iter=1000)

#entrenamiento
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)


print(confusion_matrix(y_test,predictions))

#print('\n')

print(classification_report(y_test,predictions))




