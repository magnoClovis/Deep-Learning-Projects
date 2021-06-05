# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 23:20:08 2021

@author: clovi
"""


# Import module
from tkinter import *
  
# Create object
root = Tk()
  
# Adjust size
root.geometry( "200x200" )
  
# Change the label text
def show():
    label.config( text = clicked.get() )
  
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
clicked = StringVar()


# initial menu text
clicked.set( "Monday" )
  
def change_dropdown(*args):
    global dropdown
    dropdown = str(clicked.get())
    print(dropdown)
    
# Create Dropdown menu
drop = OptionMenu( root , clicked , *options )
drop.pack()


# Create button, it will change label text
button = Button( root , text = "OK" , command = root.destroy).pack(pady=20)

# Create Label
label = Label( root , text = " " )
label.pack()
  
  
clicked.trace('w', change_dropdown)
# Execute tkinter
root.mainloop()