# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:30:41 2021

@author: clovi
"""

# 1 - DATA PREPROCESSING  

# Libraries
import numpy as np
import pandas as pd

# Importing the training set
df = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = df.iloc[:, 1:2].values

# Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_sc = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
xtrain = []
ytrain = []

for i in range(60, 1258):
    xtrain.append(training_set_sc[i-60:i,0])
    ytrain.append(training_set_sc[i, 0])
    
xtrain,ytrain = np.array(xtrain), np.array(ytrain)

# Reshaping
batch = xtrain.shape[0]
timesteps = xtrain.shape[1]
xtrain = np.reshape(xtrain, (batch, timesteps, 1))

# 2 - BUILDING THE RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (timesteps, 1)))
regressor.add(Dropout(rate=0.2))

# Adding the second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fittin the RNN to the training set
regressor.fit(xtrain, ytrain, batch_size = 32, epochs = 100)

# 3 - MAKING PREDICTIONS AND VISUALISING THE RESULTS

# Getting the real stock price of 2017
df_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = df_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
df_total = pd.concat((df['Open'], df_test['Open']), axis = 0)
inputs = df_total[len(df_total)-len(df_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

xtest = []
for i in range(60, 80):
    xtest.append(inputs[i-60:i,0])
xtest = np.array(xtest)

batch = xtest.shape[0]
timesteps = xtest.shape[1]
xtest = np.reshape(xtest, (batch, timesteps, 1))

predictions = regressor.predict(xtest)
predictions = sc.inverse_transform(predictions)

# Visualising the results
import matplotlib.pyplot as plt
plt.plot(test_set, color = 'green', label = 'Real Stock Price')
plt.plot(predictions, color = 'red', label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Proce')
plt.legend()
plt.show()

# Evaluating
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_set, predictions))
relative_error = rmse/df_test['Open'].max()
