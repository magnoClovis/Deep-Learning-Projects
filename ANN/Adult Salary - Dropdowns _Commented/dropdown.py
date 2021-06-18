# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:25:58 2021

@author: clovi
"""

'''
Here the interaction boxes for data input are created
'''

from tkinter import *
import options
import tkinter as tk
from tkinter import simpledialog

workclass, education, marital, occupation, relationship, race, sex, country = options.clean(options.get_options)


'''
This function creates the scratch of the boxes, the other functions add elements
in this one to change its appearence, they are also responsible to storing the 
input data in different variables
'''

def create_root():
    # Create object for dropdown
    global root 
    root = Tk()
   
    app_width = 250
    app_height = 250
    
    screen_width  = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    x = int((screen_width/2) - (app_width/2))
    y = int((screen_height/2) - (app_height/2))
    
    # Size
    root.geometry(f'{app_width}x{app_height}+{x}+{y}')
    
    # Datatype of menu text
    global selected 
    selected = StringVar()


# Saving the selected option
def change_dropdown(*args):
    global dropdown
    dropdown = str(selected.get())

def drop_workclass():
    create_root()
    # Create Label
    label = Label(root, text = "Workclass\n" )
    label.pack()
    
    # Initial menu text
    selected.set( "Select an Option" )
      
    # Create Dropdown menu
    drop = OptionMenu(root, selected, *workclass)
    drop.pack()
    
    # Create button, it will change label text
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)
    # Execute tkinter
    root.mainloop()
    return dropdown

def drop_education():
    create_root()
    label = Label(root, text = "Education\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *education)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_marital():
    create_root()
    label = Label(root, text = "Marital Status\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *marital)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_occupation():
    create_root()
    label = Label(root, text = "Occupation\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *occupation)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_relationship():
    create_root()
    label = Label(root, text = "Relationship\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *relationship)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_race():
    create_root()
    label = Label(root, text = "Race\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *race)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown


def drop_sex():
    create_root()
    label = Label(root, text = "Sex\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *sex)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_race():
    create_root()
    label = Label(root, text = "Race\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *race)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_country():
    create_root()
    label = Label(root, text = "Country\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, *country)
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def drop_exit(bool_predict, prob_predict):
    create_root()
    bool_text = ("\nSalary > 50K?   {}".format(bool_predict))
    prob_text = ('Probability = {}%'.format(prob_predict))
    label = Label(root, text = bool_text)
    label.pack()
    
    label = Label(root, text = prob_text)
    label.pack()
    
    label = Label(root, text = "Exit the program?\n")
    label.pack()
    
    selected.set("Select an Option")

    drop = OptionMenu(root, selected, "Yes", "No")
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown

def get_age():
    create_root()
    label = Label(root, text = "Age")
    label.pack()
    e = Entry(root,width = 40, bg = 'lightgreen', fg = 'black', borderwidth = 4)
    e.pack()
    def myClick():
        global val
        val = e.get()
        root.destroy()
        
    myButton = Button(root, text="OK", command=myClick)
    myButton.pack()
    root.mainloop()
    return int(val)

def get_ednum():
    create_root()
    label = Label(root, text = "Education Number")
    label.pack()
    e = Entry(root,width = 40, bg = 'lightgreen', fg = 'black', borderwidth = 4)
    e.pack()
    def myClick():
        global val
        val = e.get()
        root.destroy()
        
    myButton = Button(root, text="OK", command=myClick)
    myButton.pack()
    root.mainloop()
    return int(val)


def get_gain():
    create_root()
    label = Label(root, text = "Capital Gain")
    label.pack()
    e = Entry(root,width = 40, bg = 'lightgreen', fg = 'black', borderwidth = 4)
    e.pack()
    def myClick():
        global val
        val = e.get()
        root.destroy()
        
    myButton = Button(root, text="OK", command=myClick)
    myButton.pack()
    root.mainloop()
    return int(val)

def get_loss():
    create_root()
    label = Label(root, text = "Capital Loss")
    label.pack()
    e = Entry(root,width = 40, bg = 'lightgreen', fg = 'black', borderwidth = 4)
    e.pack()
    def myClick():
        global val
        val = e.get()
        root.destroy()
        
    myButton = Button(root, text="OK", command=myClick)
    myButton.pack()
    root.mainloop()
    return int(val)

def get_hours():
    create_root()
    label = Label(root, text = "Hours per Week")
    label.pack()
    e = Entry(root,width = 40, bg = 'lightgreen', fg = 'black', borderwidth = 4)
    e.pack()
    def myClick():
        global val
        val = e.get()
        root.destroy()
        
    myButton = Button(root, text="OK", command=myClick)
    myButton.pack()
    root.mainloop()
    return int(val)


# workclass_op = drop_workclass()
# education_op = drop_education()
# marital_op = drop_marital()
# occupation_op = drop_occupation()
# relationship_op = drop_relationship()
# race_op = drop_race()
# sex_op = drop_sex()
# country_op = drop_country()
# exit_op = drop_exit()

