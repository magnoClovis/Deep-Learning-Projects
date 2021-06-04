# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:26:41 2021

@author: clovi
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import os
from keras.layers import Dropout

# IMPORTING THE DATASET AND SPLITTING X AND Y

df = pd.read_csv('adult.csv')
x_1 = df.iloc[:, 0:2]
x_2 = df.iloc[:,3:-1]
x = pd.concat([x_1,x_2], axis = 1).values
y = df.iloc[: , -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 8] = le.fit_transform(x[:, 8])
y = le.fit_transform(y)


#------------------------------------------------------------------------------
''' Creating dummies variables and dictionaries to know the encoding used by 
the computer in OneHotEncoder, the dictionaries will make it easier to make
predictions when new values are inputted into the network 
'''
import encoding_keys

work_dict = encoding_keys.workclass(df)
education_dict = encoding_keys.education(df)
marital_dict = encoding_keys.marital_status(df)
occupation_dict = encoding_keys.occupation(df)
relationship_dict = encoding_keys.relationship(df)
race_dict = encoding_keys.race(df)
country_dict = encoding_keys.country(df)
#------------------------------------------------------------------------------


# ENCODING VARIABLES
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1,2,4])], remainder = 'passthrough')
ct_x1 = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [5,6,7])], remainder = 'passthrough')
ct_x2 = np.array(ct.fit_transform(x))

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [35])], remainder = 'passthrough')
ct_x3 = np.array(ct.fit_transform(ct_x2))
ct_x3 = ct_x3[:,0:42]

x_n = x[:, 0:4:3]
x_n2 = x[:, 8:12]
x_n3 = np.concatenate((x_n,x_n2),axis=1)

x1= ct_x1[:, 0:32]
x2 = ct_x2[:, 0:26]
x3 = ct_x3
x4 = np.concatenate((x1,x2),axis=1)
x5 = np.concatenate((x4,x3),axis=1)

x = np.concatenate((x5,x_n3),axis=1)

# SPLITTING INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# STANDARDIZATION 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

# BUILDING THE NEURAL NETWORK
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units =30, activation = 'relu'))
ann.add(Dropout(rate = 0.1))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(Dropout(rate = 0.1))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'sigmoid'))
ann.add(Dropout(rate = 0.1))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'sigmoid'))
ann.add(Dropout(rate = 0.1))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#-----------------------------------------------------------------------------

# LOADING WEIGHTS FROM CHECKPOINTS AND APPLYING THEM TO THE ACTUAL MODEL
checkpoint_path = "D:\clovi\Estudos\Deep-Learning-Projects\Adult Salary\cp.ckpt"
ann.load_weights(checkpoint_path)
loss,acc = ann.evaluate(xtest, ytest)
print("Restored model, accuracy:{:5.2f}%".format(100*acc))
print("Restored model, loss:{:5.2f}".format(loss))

ypred = ann.predict(xtest)
ypredb = (ypred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, ypredb)

mean_test = accuracy_score(ytest,ypredb)

#New predictions with the actual model
import new_predictions
predictions = new_predictions.NewPredictions(ann, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict)


# LOADING AND USING A PREVIOUSLY TRAINED MODEL
new_ann = tf.keras.models.load_model('adult.h5')
new_ann.summary()

ypred = new_ann.predict(xtest)
ypredb = (ypred > 0.5)ç

mean_test = accuracy_score(ytest,ypredb)

#New predictions with loaded model
predictions = new_predictions.NewPredictions(new_ann, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict)


