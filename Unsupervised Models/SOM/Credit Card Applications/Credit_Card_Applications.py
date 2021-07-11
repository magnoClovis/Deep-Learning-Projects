# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:36:41 2021

@author: clovi
"""

'''
The purpose of this algorithm is to generate a Self-Organizing Map of customers
and detect frauds on applications that customers have presented to upgrade their 
credit card. Using SOM, the purpose is the model to identify patterns on the applications
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
The file concerns credit card applications. All atribute names and avalues have been changed
to meaningless symbols to protect confidentiality of the data.
This dataset is interesting becaus there is a good mix of attributes -- 
continuous, nominal with smal numbers of values, and nominal with larger numbers of values. 
There are also a few missing values
'''

# Importing the dataset
df = pd.read_csv('Credit_Card_Applications.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

'''
To work on this data, what is going to be done is, first, identify the input (the customers)
and for each customer is initialized with a vector of weights, therefore the output is 
the most similar neuron to the customer. Then the weights of the neighbors neurons are 
updated according to the weight generated in the last step. The process will repeat with 
all the customers.
'''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0,
              learning_rate = 0.5)
'''
x and y are the dimension for the final map (in this case, 10x10)
input_len -- number of features in the dataframe, more specifically, in X
sigma -- radius of the neighborhood in the grid
learning_rate -- how much the weights are updated in each observation, the higher, the faster
decay_function -- can be used to improve the convergence
'''

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
frauds = mappings[(8,3)]
frauds = sc.inverse_transform(frauds)
