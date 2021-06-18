# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:11:51 2021

@author: clovi
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# IMPORTING THE DATASET 
df = pd.read_excel('Folds5x2_pp.xlsx')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# BUILDING THE ANN
ann = tf.keras.Sequential()

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 1))

# TRAINING THE ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# TRAINING
ann.fit(xtrain,ytrain, batch_size = 32, epochs = 100)

# PREDICTING THE RESULTS OF THE TEST SET
ypred = ann.predict(xtest)
np.set_printoptions(precision=2)

ypred_ytest = np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)), 1)
print(ypred_ytest)
