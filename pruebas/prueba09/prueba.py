###############################################################################

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
from keras.layers import Dense, TimeDistributed, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

###############################################################################

# archivo de datos
ARCHIVO_CSV = 'dataset.csv'

###############################################################################

# recuperar datos
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Cargando datos desde ' + ARCHIVO_CSV + '...')
data = pd.read_csv(ARCHIVO_CSV, sep=',', header=0)

# extraer columna a predecir
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Extrayendo columnas...')
y = data.b5
X = data.drop('b5', axis=1)

# normalizar (?)
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Normalizando X...')
#X = StandardScaler().fit_transform(X)
X = to_categorical(X)
###y = to_categorical(y)

# separar training set y validation set
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Separando datasets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


def crear_modelo_rnn_lstm():
	#model = Sequential()
	#model.add(LSTM(128, input_shape=(1, 64)))
	#model.add(Dense(256, activation='relu'))
	#model.add(Dense(75, activation='softmax'))
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#return model


	#model = Sequential()
	#model.add(Embedding(5 + 1, 128, input_length=5, dropout=0.2))
	#model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
	#model.add(Dense(5))
	#model.add(Activation('softmax'))
	#model.compile(loss='binary_crossentropy', optimizer='rmsprop')
	#return model

	model = Sequential()
	#model.add(LSTM(512, input_shape=(5, 256), return_sequences=True))
	model.add(LSTM(256, input_shape=(5, 256)))
	#model.add(LSTM(512))
	#model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
clasificador = KerasClassifier(build_fn=crear_modelo_rnn_lstm, epochs=400, batch_size=2048, verbose=2)

print()
print('=' * 80)
print(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S.%f") + ': Iniciando...')

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