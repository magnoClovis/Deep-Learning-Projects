# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 02:02:39 2021

@author: clovi
"""

# Part 1 - Identify the Frauds with the Self-Organizing Map

'''
This part is about making the unsipervised deep learning branch of the hybrid DL model
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Credit_Card_Applications.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0,
              learning_rate = 0.5)


som.random_weights_init(x)
som.train_random(data = x , num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x1 in enumerate(x):
    w = som.winner(x1)
    plot(w[0]+0.5,
         w[1]+0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 1.5)
show()    

# Finding the frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(7,8)], mappings[(1,4)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating tue matrix of features
customers = df.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(df))
for i in range(len(df)):
    if df.iloc[i, 0] in frauds:
        is_fraud[i] = 1
    

# Scaling
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
ss = StandardScaler()
customers = ss.fit_transform(customers)
 
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Input layer and the first hideen layer
ann.add(tf.keras.layers.Dense(units = 2, kernel_initializer = 'uniform', activation='relu', input_dim = 15))

# Output layer
ann.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation='sigmoid'))

# Compiling 
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
ann.fit(customers, is_fraud, batch_size=1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the probabilities of frauds
y_pred = ann.predict(customers)
y_pred = np.concatenate((df.iloc[:, 0:1], y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]



















