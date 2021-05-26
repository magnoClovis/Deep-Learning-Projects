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
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)


# ------------ BUILDING THE ANN LAYERS ------------

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Input layer and the first hideen layer

ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))
ann.add(Dropout(rate = 0.1))

# Second hidden layer
ann.add(tf.keras.layers.Dense(units = 12, activation='relu'))
ann.add(Dropout(rate = 0.1))

# Third hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))
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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential # serve pra iniciar a rede neural
from tensorflow.keras.layers import Dense # serve para fazer as camadas
def build_classifier():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))
    ann.add(Dropout(rate = 0.1))
    ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 10)
mean_acc = accuracies.mean()


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

a = [[0,0,420,0,60,1,12000,1,1,0,300000]]
new = ann.predict(sc.transform(a))
print(float(new))
print(bool(new > 0.5))

# ------------ FUNCTION FOR MAKING NEW PREDICTIONS ------------

def new_predictions(ann, predict =''):
    
    predictions = []
    while True:
        data_list = []
        dict_country = {'france' : (0,0), 'germany': (1,0), 'spain': (0,1)}
        dict_gender = {'male': 1, 'female': 0}
        bool_dict = {'yes': 1, 'no':0}
        country = str(input("Country: ")).lower()
        credit = int(input("Credit Score: "))
        gender = str(input("Gender: ")).lower()
        age = int(input("Age: "))
        tenure = int(input("Tenure: "))
        balance = float(input("Balance: "))
        prodc = int(input("Number of products: "))
        card = str(input("Does this customer have a credit card? ")).lower()
        active = str(input("Is this customer an active member? ")).lower()
        salary = float(input("Estimated salary: "))
        
        tpl_country = dict_country[country]
        int_gender = dict_gender[gender]
        int_card = bool_dict[card]
        int_active = bool_dict[active]
        
        data_list = [[tpl_country[0], tpl_country[1], credit, int_gender, age, tenure, balance,
                     prodc, int_card, int_active, salary]]
        
        new_predict = ann.predict(sc.transform(data_list))
        prob_predict = round((float(new_predict))*100,2)
        bool_predict = (bool(new_predict > 0.5))
        
        result = (prob_predict,bool_predict) 
        predictions.append(result)
        
        print('The probability of the customer to leave the bank is about', prob_predict, '%.')
        
        verif = str(input("Type 'exit' to exit the program or press anything to make another prediction. \n")).lower()
        if verif == 'exit':
            return predictions


predictions = new_predictions(ann)
