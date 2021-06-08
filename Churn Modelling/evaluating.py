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
from evaluating import build_classifier, evaluating_accuracy


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

def build_classifier():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units = 55, 'relu'))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = 75, 'relu'))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = 55, 'relu'))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann

classifier = KerasClassifier(build_fn = build_ann, batch_size = 32, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 10)
mean_acc = accuracies.mean()




