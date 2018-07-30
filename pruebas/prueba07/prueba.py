###############################################################################
# Experimento - entrenamiento y validación con diferentes modelos
###############################################################################
#
# Entrena diferentes modelos con el mismo dataset e imprime los resultados.
#

import time
import datetime
import os

#xxx - nogpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # no funciona
###os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier


# archivo de datos
ARCHIVO_CSV = 'dataset.csv'

# nombres de los clasificadores
nombres = [
		"Vecinos más próximos (Nearest Neighbors)",
		"Bayesiano ingenuo (Naive Bayes)",
		"SVM Lineal (Linear SVM)",
		"Árbol de decisión (Decision Tree)",
		"Bosques aleatorios (Random Forest)",
		"Red Neuronal (Neural Network, NN)",
		"NN Recurrente con LSTM (RNN with LSTM)"
	]

# clasificadores
clasificadores = [
		KNeighborsClassifier(3),
		GaussianNB(),
		SVC(kernel="linear", C=0.025),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
	]


# recuperar datos
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Cargando datos desde ' + ARCHIVO_CSV + '...')
data = pd.read_csv(ARCHIVO_CSV, sep=',', header=0)

# extraer columna a predecir
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Extrayendo columnas...')
y = data.aff
X = data.drop('aff', axis=1)

# normalizar (?)
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Normalizando X...')
X = StandardScaler().fit_transform(X)


# separar training set y validation set
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Separando datasets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
# para lstm...
X_train_lstm = X_train.reshape((-1, 1, 64)) # 64 variables
X_test_lstm = X_test.reshape((-1, 1, 64))

# funciones (básicas) para crear modelos de redes para keras, requerida por KerasClassifier
def crear_modelo_nn():
	model = Sequential()
	model.add(Dense(128, input_dim=64, activation='relu'))


	model.add(Dense(256, activation='relu'))


	#model.add(Dense(128, activation='relu'))	

	#model.add(Dense(1, activation='sigmoid'))
	model.add(Dense(75, activation='softmax'))

	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.compile(optimizer='sgd', loss='mse')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
clasificadores.append(KerasClassifier(build_fn=crear_modelo_nn, epochs=300, batch_size=4096, verbose=2))

def crear_modelo_rnn_lstm():
	model = Sequential()
	model.add(LSTM(128, input_shape=(1, 64)))

	model.add(Dense(256, activation='relu'))

	model.add(Dense(75, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
clasificadores.append(KerasClassifier(build_fn=crear_modelo_rnn_lstm, epochs=300, batch_size=4096, verbose=2))


# proceso principal
for nombre, clasificador in zip(nombres, clasificadores):

	#xxx
	if not 'NN' in nombre:
		continue

	print()
	print('=' * 80)
	print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Iniciando prueba para ' + nombre)

	if 'RNN' in nombre:
		clasificador.fit(X_train_lstm, y_train)
		prediccion = clasificador.predict(X_test_lstm)
	else:
		clasificador.fit(X_train, y_train)
		prediccion = clasificador.predict(X_test)
	score = accuracy_score(y_test, prediccion)
	matriz_confusion = confusion_matrix(y_test, prediccion)	
	reporte = classification_report(y_test, prediccion)
	print('Score general: ' + str(score))
	print('Matríz de confusión:')	
	print(matriz_confusion)
	print("Reporte de resultados:")
	print(reporte)
	print('#' * 80)
