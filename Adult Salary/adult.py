# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:32:55 2021

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
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [41])], remainder = 'passthrough')
ct_x3 = np.array(ct.fit_transform(ct_x1))

x_n = x[:, 0:4:3]
x_n2 = x[:, 8:12]
x_n3 = np.concatenate((x_n,x_n2),axis=1)

x1= ct_x1[:, 0:31]
x2 = ct_x2[:, 0:26]
x3 = ct_x3[:, 0:74]
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



#------------------------------------------------------------------------------
# CHECKPOINTS

#Create checkpoint path

checkpoint_path = "D:\clovi\Estudos\Deep-Learning-Projects\Adult Salary\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
hist = ann.fit(xtrain, ytrain, batch_size = 32, epochs = 80, callbacks = [cp_callback])
#------------------------------------------------------------------------------



hist = ann.fit(xtrain, ytrain, batch_size = 32, epochs = 80)

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

from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()

def NewPredictions(ann, ss, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict):
    
    predictions = []
    while True:
        data_list = []
        gender_dict = {'male': 1, 'female': 0}
        age = int(input("Age: "))
        workclass = str(input("Workclass: ")).replace("-","").replace(" ","").lower()
        education = str(input("Education: ")).replace("-","").replace(" ","").lower()
        education_num = int(input("Education number: "))
        marital = str(input("Marital status: ")).replace("-","").replace(" ","").lower()
        occupation = str(input("Occupation: ")).replace("-","").replace(" ","").lower()
        relationship = str(input("Relationship: ")).replace("-","").replace(" ","").lower()
        race = str(input("Race: ")).replace("-","").replace(" ","").lower()
        gender = str(input("Gender: ")).replace("-","").replace(" ","").lower()
        capital_gain = int(input("Capital gain: "))
        capital_loss = int(input("Capital loss: "))
        hours = int(input("Hours per week: "))
        education = int(input("Credit Score: "))
        country = str(input("Native country: ")).replace("-","").replace(" ","").lower()
         
        
        workclass_lst = work_dict[workclass]
        education_lst = education_dict[education]
        marital_lst = marital_dict[marital]
        occupation_lst = occupation_dict[occupation]
        relationship_lst = relationship_dict[relationship]
        race_lst = race_dict[race]
        country_lst = country_dict[country]
        gender_int = gender_dict[gender]


        '''Order of the data: Workclass, Education, Marital Status, Occupation
        Relationship, Race, Native Country, Age, Education Number, Gender,
        Capital Gain, Capital Loss, Hours per week'''
        
        
        for j in range(len(cu_list)):
            data_list = [i for i in cu_list[j]]
            
            
        data_list = [i for i in workclass_lst]
        new_predict = ann.predict(ss.transform(data_list))
        prob_predict = round((float(new_predict))*100,2)
        bool_predict = (bool(new_predict > 0.5))
        

        
        result = (prob_predict,bool_predict) 
        predictions.append(result)
        
        print('The probability of the customer to leave the bank is about', prob_predict, '%.')
        
        verif = str(input("Type 'exit' to exit the program or press anything to make another prediction. \n")).lower()
        if verif == 'exit':
            return predictions


'''
The code below stands for getting the checkpoints and saved models if necessary

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(xtest, ytest)
print("Restored model, accuracy:{:5.2f}%".format(100*acc))


new_model = keras.models.load_model('my_model.h5')
new_model.summary()
'''

