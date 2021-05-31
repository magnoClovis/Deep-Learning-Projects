# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:43:58 2021

@author: clovi
"""
import tensorflow as tf
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 



def build_classifier(unit_1, unit_2, unit_3, actv_1 = 'relu', actv_2 = 'relu', actv_3 = 'relu',dropout=0.1):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units = unit_1, activation=actv_1))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = unit_2, activation=actv_2))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = unit_3, activation=actv_3))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann

def evaluating_accuracy(ss, xtrain, ytrain, num_batch, num_epochs, num_cv, build_ann):

    classifier = KerasClassifier(build_fn = build_ann, batch_size = num_batch, epochs = num_epochs)
    accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = num_cv)
    mean_acc = accuracies.mean()
    
    return mean_acc



