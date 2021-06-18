# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:32:55 2021

@author: clovi
"""

'''
Here is where the code will execute the training of the network if it has not 
been trained, and also it is where it's called the "NewPredictions" function 
that therefore calls the other fuinctions, enabling the program to deal with new 
data and make new predictions based on the pre trained net.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from keras.layers import Dropout



# The function below is responsible for the data preprocessing
def data_preprocessing(df):
    # Selecting the columns of the data set with relevant data
    x_1 = df.iloc[:, 0:2]
    x_2 = df.iloc[:,3:-1]
    x = pd.concat([x_1,x_2], axis = 1).values
    y = df.iloc[: , -1].values
    
    # Label Encoding one of the columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    x[:, 8] = le.fit_transform(x[:, 8])
    y = le.fit_transform(y)
    
    
    #------------------------------------------------------------------------------
    ''' Creating dummy variables and dictionaries to know the encoding used by 
    the computer in OneHotEncoder. The dictionaries will make it easier to make
    predictions when new values are inputted into the network .
    
    Here it is used some functions inside "encoding_keys.py" that I developed to 
    execute the task of coding and saving the coded values into dictionaries.
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
    
    '''
    As a dictionary with the encoding used is already created, it is now possible 
    to encode the data and substitute the values for the encoded ones.
    '''
    
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
    return xtrain, xtest, ytrain, ytest, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict


# The function below is responsible to construct and train the ann model
def building_model(xtrain, xtest, ytrain, ytest, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict):
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
    
    
    #------------------------------------------------------------------------------
    # CHECKPOINTS
    
    #Create checkpoint path
    
    checkpoint_path = "D:\clovi\Estudos\Deep-Learning-Projects\Adult Salary - Dropdowns\cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    #Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
    hist = ann.fit(xtrain, ytrain, batch_size = 32, epochs = 80, callbacks = [cp_callback])
    #------------------------------------------------------------------------------
    
    ann.save('adult.h5') 
    
    mean = np.mean(hist.history['accuracy'])
    
    # EVALUATING THE NETWORK
    ypred = ann.predict(xtest)
    ypredb = (ypred > 0.5)
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(ytest, ypredb)
    
    mean_test = accuracy_score(ytest,ypredb)
    loss = np.mean(hist.history['loss'])
    
    # NEW PREDICTIONS 
    
    import new_predictions
    predictions = new_predictions.NewPredictions(ann, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict)
    
    return predictions

# The function below is responsible to load some already existing trained model
def load_model(sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict):

    import keras
    ann = keras.models.load_model('./adult.h5')

    import new_predictions
    predictions = new_predictions.NewPredictions(ann, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict)
    
    return predictions


if __name__ == '__main__':
    
    file = os.path.isfile('./adult.h5')
    df = pd.read_csv('adult.csv')
    
    xtrain, xtest, ytrain, ytest, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict = data_preprocessing(df)
   
    if file == False:
        predictions = building_model(xtrain, xtest, ytrain, ytest, sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict)
    
    else:
        predictions = load_model(sc, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict)
    

'''
The code below stands for getting the checkpoints and saved models if necessary

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(xtest, ytest)
print("Restored model, accuracy:{:5.2f}%".format(100*acc))


new_model = keras.models.load_model('my_model.h5')
new_model.summary()
'''

