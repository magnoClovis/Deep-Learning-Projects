# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:32:55 2021

@author: clovi
"""

import pandas as pd
import numpy as np
import tensorflow as tf

# IMPORTAR DATASET E SEPARAR EM VARIAVEL DEPENDENTE E INDEPENDENTE

df = pd.read_csv('adult.csv')
x_1 = df.iloc[:, 0:2]
x_2 = df.iloc[:,3:-1]
x = pd.concat([x_1,x_2], axis = 1).values
y = df.iloc[: , -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 8] = le.fit_transform(x[:, 8])
y = le.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1,2,4])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [34,35,36,41])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# SEPARANDO ENTRE CONJUNTO DE TREINO E CONJUNTO DE TESTE
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# PADRONIZAR OS VALORES
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

# CONSTRUINDO A REDE NEURAL

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = , activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 30, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 30, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

hist = ann.fit(xtrain, ytrain, batch_size = 25, epochs = 100)

mean = np.mean(hist.history['accuracy'])

# AVALIANDO A REDE
ypred = ann.predict(xtest)
ypredb = (ypred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, ypredb)

mean_test = accuracy_score(ytest,ypredb)
loss = np.mean(hist.history['loss'])