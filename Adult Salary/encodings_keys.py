# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 01:16:27 2021

@author: clovi
"""

def workclass(df):
    instance = tuple(df['workclass'].unique())
    instance_df = pd.DataFrame(instance, columns=['workclass'])
    dum_df = pd.get_dummies(instance_df, columns = ['workclass'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    work_dict = {}
    for i in range(len(instance_keys)):
        work_dict[instance_keys[i]] = instance_values[i]
    
    return work_dict



def education(df):
    instance = tuple(df['education'].unique())
    instance_df = pd.DataFrame(instance, columns=['education'])
    dum_df = pd.get_dummies(instance_df, columns = ['education'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    education_dict = {}
    for i in range(len(instance_keys)):
        education_dict[instance_keys[i]] = instance_values[i]
    
    return education_dict



def marital_status(df):
    instance = tuple(df['marital-status'].unique())
    instance_df = pd.DataFrame(instance, columns=['marital-status'])
    dum_df = pd.get_dummies(instance_df, columns = ['marital-status'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    marital_dict = {}
    for i in range(len(instance_keys)):
        marital_dict[instance_keys[i]] = instance_values[i]
    
    return marital_dict


def occupation(df):
    instance = tuple(df['occupation'].unique())
    instance_df = pd.DataFrame(instance, columns=['occupation'])
    dum_df = pd.get_dummies(instance_df, columns = ['occupation'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    occupation_dict = {}
    for i in range(len(instance_keys)):
        occupation_dict[instance_keys[i]] = instance_values[i]
    
    return occupation_dict


def relationship(df):
    instance = tuple(df['relationship'].unique())
    instance_df = pd.DataFrame(instance, columns=['relationship'])
    dum_df = pd.get_dummies(instance_df, columns = ['relationship'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    relationship_dict = {}
    for i in range(len(instance_keys)):
        relationship_dict[instance_keys[i]] = instance_values[i]
    
    return relationship_dict


def race(df):
    instance = tuple(df['race'].unique())
    instance_df = pd.DataFrame(instance, columns=['race'])
    dum_df = pd.get_dummies(instance_df, columns = ['race'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    race_dict = {}
    for i in range(len(instance_keys)):
        race_dict[instance_keys[i]] = instance_values[i]
    
    return race_dict



def country(df):
    instance = tuple(df['native-country'].unique())
    instance_df = pd.DataFrame(instance, columns=['native-country'])
    dum_df = pd.get_dummies(instance_df, columns = ['native-country'])
    instance_df = instance_df.join(dum_df)
    instance_list = instance_df.values.tolist()
    
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)
    
    instance_values = []
    for i in range(len(instance_list)):
        code = instance_list[i][1:]
        instance_values.append(code)
        
    country_dict = {}
    for i in range(len(instance_keys)):
        country_dict[instance_keys[i]] = instance_values[i]
    
    return country_dict
