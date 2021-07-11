# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:30:15 2021

@author: clovi
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8 # sequence length
D = 2 # input dimensionality
M = 3 # hidden layers size

X = np.random.randn(1, T, D) # dummy input data (batch size, sequence length, feature dimensionality)

def lstm1():
    input_ = Input(shape=(T, D)) # create input of size TxD, works as a single sentece in this case
    rnn = LSTM(M, return_state = True) # creating LSTM layer
    x = rnn(input_) 
    
    model = Model(inputs=input_, outputs = x) # Model(???)
    o, h, c = model.predict(X)
    print("o:", o) # actual output
    print("h:", h) # hidden state
    print("c:", c) # cell state
    
def lstm2():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state = True, return_sequences = True) # retunr_sequences = True: what sequences are returned and how it relates to what happens en it is false
    # rnn = GRU(M, return_state = True)
    x = rnn(input_)
    
    model = Model(inputs=input_, outputs = x)
    o, h, c = model.predict(X)
    print("o:", o)
    print("h:", h)
    print("c:", c)
    
def gru1():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state = True)
    x = rnn(input_)
    
    model = Model(inputs=input_, outputs = x)
    o, h = model.predict(X)
    print("o:", o)
    print("h:", h)
    
def gru2():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state = True, return_sequences = True)
    x = rnn(input_)
    
    model = Model(inputs=input_, outputs = x)
    o, h = model.predict(X)
    print("o:", o)
    print("h:", h)
    

print('lstm1:')
lstm1()
print('lstm2:')
lstm2()    
print('gru1:')
gru1()  
print('gru2:')
gru2()      
    
    
'''
OUTPUTS: (may vary)
lstm1:
o: [[ 0.1275934  -0.06007639  0.12466393]]
h: [[ 0.1275934  -0.06007639  0.12466393]]
c: [[ 0.25404194 -0.091059    0.3132385 ]]
Here where states are returned, but sequences are not, it is possible to see that 
'o' and 'h' have the same values. (The output is the same as one of the hidden layers).



lstm2:
o: [[[ 0.0366511  -0.01497604 -0.00305384]
  [ 0.0238916   0.00626355  0.01306375]
  [ 0.09402706  0.02095795  0.04927247]
  [ 0.12088887  0.0146521   0.04389957]
  [ 0.02428926 -0.02125001 -0.03280836]
  [-0.18480785  0.07955368  0.09650832]
  [-0.1169995   0.14709108  0.17059399]
  [-0.0854052  -0.00965337  0.06430553]]]
h: [[-0.0854052  -0.00965337  0.06430553]]
c: [[-0.18330091 -0.01728494  0.09713234]]
Here where the sequences are returned, it is possible to observe that 'h' is equal
to the last value of 'o', in other words, 'h' and 'c'represent the final hidden states
of the LSTM, they are de hidden state and the cell state at the final timestep of the input.



gru1:
o: [[-0.10308884  0.00602481 -0.19637299]]
h: [[-0.10308884  0.00602481 -0.19637299]]
Here same thing that happened in lstm1 happens.



gru2:
o: [[[ 0.14022498  0.14378056  0.10197645]
  [ 0.05356785  0.01429433 -0.01710364]
  [ 0.32864082  0.15917574  0.15899462]
  [ 0.37033767  0.23339218  0.13465783]
  [ 0.11087506 -0.08772575 -0.03266219]
  [-0.4439042  -0.42803743 -0.29885212]
  [-0.26571584 -0.4089483  -0.15209348]
  [-0.22992311  0.13136417  0.03558518]]]
h: [[-0.22992311  0.13136417  0.03558518]]
Here same thing that happened in lstm2 happens.

'''
    
    
    
    
    
    
    
    