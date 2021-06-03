# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:32:55 2021

@author: clovi
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os

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

#----
# Creating dummies variables to know the encoding used by the computer 

import encoding_keys

work_dict = encoding_keys.workclass(df)
education_dict = encoding_keys.education(df)
marital_dict = encoding_keys.marital_status(df)
occupation_dict = encoding_keys.occupation(df)
relationship_dict = encoding_keys.relationship(df)
race_dict = encoding_keys.race(df)
country_dict = encoding_keys.country(df)
#----


# ENCODING VARIABLES
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1,2,4])], remainder = 'passthrough')
x_1 = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [5,6,7])], remainder = 'passthrough')
x_2 = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [41])], remainder = 'passthrough')
x_3 = np.array(ct.fit_transform(x_1))

x_n = x[:, 8:12]
x_n2 = x[:, 0:4:3]
x_n3 = np.concatenate((x_n2,x_n),axis=1)

x1= x_1[:, 0:31]
x2 = x_2[:, 0:26]
x3 = x_3[:, 0:74]
x4 = np.concatenate((x1,x2),axis=1)
x5 = np.concatenate((x4,x3),axis=1)

x = np.concatenate((x5,x_n3),axis=1)

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
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Create checkpoint path
checkpoint_path = "D:\clovi\Estudos\Deep-Learning-Projects\Adult Salary\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)

hist = ann.fit(xtrain, ytrain, batch_size = 25, epochs = 200, callbacks = [cp_callback])

ann.save('adult.h5')

mean = np.mean(hist.history['accuracy'])

# AVALIANDO A REDE
ypred = ann.predict(xtest)
ypredb = (ypred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, ypredb)

mean_test = accuracy_score(ytest,ypredb)
loss = np.mean(hist.history['loss'])

'''
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(xtest, ytest)
print("Restored model, accuracy:{:5.2f}%".format(100*acc))


new_model = keras.models.load_model('my_model.h5')
new_model.summary()
'''
