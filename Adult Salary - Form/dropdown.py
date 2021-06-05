# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:25:58 2021

@author: clovi
"""
from tkinter import *
import options

workclass, education, marital, occupation, relationship, race, sex, country = options.clean(options.get_options)

def create_root():
    # Create object for dropdown
    global root 
    root = Tk()
      
    # Size
    root.geometry("200x200")
    
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
    label = Label(root, text = "Workclass\n")
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

def drop_exit():
    create_root()
    label = Label(root, text = "Exit the program?\n")
    label.pack()
    
    selected.set( "Select an Option" )

    drop = OptionMenu(root, selected, "Yes", "No")
    drop.pack()
    
    button = Button(root, text = "OK", command = root.destroy).pack(pady=20)
      
    selected.trace('w', change_dropdown)

    root.mainloop()
    return dropdown


workclass_op = drop_workclass()
education_op = drop_education()
marital_op = drop_marital()
occupation_op = drop_occupation()
relationship_op = drop_relationship()
race_op = drop_race()
sex_op = drop_sex()
country_op = drop_country()
exit_op = drop_exit()