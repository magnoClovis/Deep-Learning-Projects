# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 23:20:08 2021

@author: clovi
"""

'''
Here are the funciton responsible to get the inputs from the boxes, these functions
also standarize some input data when it is needed
'''

import pandas as pd
import numpy as np

df = pd.read_csv('adult.csv')


# Getting the possible options

def get_options(df):
    workclass = list(df['workclass'].unique())
    workclass.remove(' ?')
    workclass.append(' Other')
    
    education = list(df['education'].unique())
    
    marital = list(df['marital-status'].unique())

    occupation = list(df['occupation'].unique())
    occupation.remove(' ?')
    occupation.append(' Other')
    
    relationship = list(df['relationship'].unique())
    
    race = list(df['race'].unique())
    
    sex = list(df['sex'].unique())
    
    country = list(df['native-country'].unique())
    country.remove(' ?')
    country.append(' Other')

    return (workclass, education, marital, occupation, relationship, race, sex, country)

def clean(get_options):
    workclass=[]
    education=[]
    marital=[]
    occupation=[]
    relationship=[]
    race=[]
    sex=[]
    country=[]
    
    classes = {0:workclass, 1:education, 2:marital, 3:occupation, 4:relationship, 5:race, 6:sex, 7:country}
    
    all_options = get_options(df)
    for i in range(len(all_options)):
        for j in range(len(all_options[i])):
            options_std = (all_options[i][j].replace("-"," "))
            classes[i].append(options_std)
            
    return workclass, education, marital, occupation, relationship, race, sex, country

workclass, education, marital, occupation, relationship, race, sex, country = clean(get_options)