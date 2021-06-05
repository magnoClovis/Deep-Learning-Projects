# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 23:20:08 2021

@author: clovi
"""


# Import module
from tkinter import *
import pandas as pd
import numpy as np

df = pd.read_csv('adult.csv')


# Getting possible options

def get_options(df):
    workclass = list(df['workclass'].unique())
    workclass.remove(' ?')
    workclass.append('Other')
    
    education = list(df['education'].unique())
    
    marital = list(df['marital-status'].unique())

    occupation = list(df['occupation'].unique())
    occupation.remove(' ?')
    occupation.append('Other')
    
    relationship = list(df['relationship'].unique())
    
    race = list(df['race'].unique())
    
    sex = list(df['sex'].unique())
    
    country = list(df['native-country'].unique())
    country.remove(' ?')
    country.append('Other')
    

    return (workclass, education, marital, occupation, relationship, race, sex, country)

all_options = get_options(df)
all_options_list=[]  
    instance_keys = []
    for i in range(len(instance_list)):
        instance_std = ((instance_list[i][0].replace("-","")).replace(" ","")).lower()
        instance_keys.append(instance_std)

# Create object
root = Tk()
  
# Adjust size
root.geometry("200x200")
  
# Change the label text
def show():
    label.config(text = clicked.get())
  
# Dropdown menu options
options = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]
  
# datatype of menu text
selected = StringVar()


# initial menu text
selected.set( "Monday" )
  
def change_dropdown(*args):
    global dropdown
    dropdown = str(selected.get())
    
# Create Dropdown menu
drop = OptionMenu(root, selected, *workclass)
drop.pack()


# Create button, it will change label text
button = Button(root, text = "OK", command = root.destroy).pack(pady=20)

# Create Label
label = Label(root, text = " ")
label.pack()
  
  
selected.trace('w', change_dropdown)
# Execute tkinter
root.mainloop()