# -*- coding: utf-8 -*-
"""
Created on Wed May 26 08:02:26 2021

@author: clovi
"""

# Importing the libraries
import numpy as np
import pandas as pd 
import tensorflow as tf
from keras.layers import Dropout

# ------------ DATA PREPROCESSING ------------

# Importing the dataset
df = pd.read_csv('Churn_Modelling.csv')
x = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values 

# Encoding variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
x = x[:,1:]

# Splitting into training set and test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
xtrain = ss.fit_transform(xtrain)
xtest = ss.fit_transform(xtest)


# ------------ BUILDING THE ANN LAYERS ------------

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Input layer and the first hideen layer

ann.add(tf.keras.layers.Dense(units = 55, activation='relu'))
ann.add(Dropout(rate = 0.1))

# Second hidden layer
ann.add(tf.keras.layers.Dense(units = 75, activation='relu'))
ann.add(Dropout(rate = 0.1))

# Third hidden layer
ann.add(tf.keras.layers.Dense(units = 55, activation='relu'))
ann.add(Dropout(rate = 0.1))

# Output layer
ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

# Compiling 
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ------------ TRAINING ------------
hist = ann.fit(xtrain, ytrain, batch_size = 32, epochs = 100)

# ------------ EVALUATING ------------

ypred = ann.predict(xtest)
ypredb = (ypred > 0.5)

train = np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, ypredb)
print(cm)

mean_test = accuracy_score(ytest,ypredb)
mean_train = np.mean(hist.history['accuracy'])
loss = np.mean(hist.history['loss'])


# ----------------------------------

from evaluating import build_classifier, evaluating_accuracy

# build_classifier(unit_1, unit_2, unit_3, actv_1 = 'relu', actv_2 = 'relu', actv_3 = 'relu',dropout=0.1)
# evaluating_accuracy(ss, xtrain, ytrain, num_batch, num_epochs, num_cv, build_ann)

build_ann = build_classifier(55, 77, 55)
mean_acc = evaluating_accuracy (ss, xtrain, ytrain, 32, 100, 10, build_ann)

# ------------NEW PREDICTIONS------------

'''
FIRST CASE
    Geography: Germany
    Credit Score: 521
    Gender: Male
    Age: 45 years old
    Tenure: 1 year
    Balance: $8000
    Number of Products: 1
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $35000


SECOND CASE
    Geography: Germany
    Credit Score: 780
    Gender: Male
    Age: 35 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 3
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000
    
THIRD CASE
    Geography: France
    Credit Score: 700
    Gender: Female
    Age: 30 years old
    Tenure: 3 years
    Balance: $62000
    Number of Products: 1
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $52000
    
FOURTH CASE
    Geography: Spain
    Credit Score: 630
    Gender: Female
    Age: 27 years old
    Tenure: 1 year
    Balance: $20000
    Number of Products: 1
    Does this customer have a credit card ? No
    Is this customer an Active Member: Yes
    Estimated Salary: $31000
'''

first = [[1,0,521,1,45,1,8000,1,1,1,35000]]

second = [[1,0,780,1,35,3,60000,3,1,1,50000]]

third = [[0,0,700,0,30,3,62000,1,1,1,52000]]

fourth = [[0,1,630,0,27,1,20000,1,0,1,31000]]

test = [[0,0,420,0,60,1,12000,1,1,0,300000]]
new = ann.predict(ss.transform(test))
print(float(new))
print(bool(new > 0.5))

# ------------ FUNCTION FOR MAKING NEW PREDICTIONS ------------

from new_predictions import NewPredictions

predictions = NewPredictions(ann, ss)

''' It is necessary to pass the StandardScaler parameter into the function becausse that way
the function can understand the fitting method used, otherwise, the NewPredictions function 
(when running from another file) returns errors,'''
