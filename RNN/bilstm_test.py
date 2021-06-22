# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:18:48 2021

@author: clovi
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np
import matplotlib.pyplot as plt

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)

input_ = Input(shape=(T,D))
rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=True))
# rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=False))
x = rnn(input_)

model = Model(inputs=input_, outputs=x)
o, h1,c1,h2,c2 = model.predict(X)
print('o:', o)
print('o.shape:', o.shape)
print('h1:', h1)
print('h2:', h2)
print('c1:', c1)
print('c2:', c2)

'''
Analising the results

o: [[[ 0.0974477   0.14274234 -0.12893109  0.07745565 -0.06276552
    0.10710937]
  [ 0.13678706  0.20848148 -0.1383167  -0.09534412  0.08618465
   -0.04994639]
  [ 0.09793569  0.08529385 -0.12219412 -0.05424634  0.06138496
    0.00944298]
  [ 0.11448028  0.1069451  -0.04896816 -0.11979487  0.09056767
   -0.0955522 ]
  [ 0.04155177 -0.04894564  0.05103248 -0.03650804  0.04309508
    0.00300818]
  [ 0.04818463  0.03509663 -0.03108103  0.0687714  -0.05496958
    0.06482491]
  [ 0.03622275  0.02017772 -0.02629804  0.00137034 -0.00180362
   -0.00069337]
  [ 0.04924886  0.03884381 -0.02123812  0.00074955 -0.00668245
   -0.00782336]]]


o.shape: (1, 8, 6)


h1: [[ 0.04924886  0.03884381 -0.02123812]]
h2: [[ 0.07745565 -0.06276552  0.10710937]]
c1: [[ 0.10325773  0.07689673 -0.04296517]]
c2: [[ 0.11238643 -0.24446164  0.44392872]]


It is possible to see that 'h1' is equal to the first three elements of the last output,
but 'h2' does not match with the last three elements of the last output. It happens because
'h2' is the final hidden state of the Bidirectional RNN, then 'h2' corresponds to the last
three elements in the first output.

'''


T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)

input_ = Input(shape=(T,D))
rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=False))
x = rnn(input_)

model = Model(inputs=input_, outputs=x)
o, h1,c1,h2,c2 = model.predict(X)
print('o:', o)
print('o.shape:', o.shape)
print('h1:', h1)
print('h2:', h2)
print('c1:', c1)
print('c2:', c2)

'''
Analising the outputs

o: [[ 0.06358749  0.12500986  0.03030911 -0.04427626  0.07919332  0.00623538]]

o.shape: (1, 6)

h1: [[0.06358749 0.12500986 0.03030911]]
h2: [[-0.04427626  0.07919332  0.00623538]]
c1: [[0.12657847 0.22591904 0.06544125]]
c2: [[-0.0799371   0.12356586  0.01738513]]

Here with retunrn_sequences = False we have that the output is a concatenation of
'h1' and 'h2', therefore, it is a concatenation of the three first elements of the last row
and the three last elements of the first row.


'''
