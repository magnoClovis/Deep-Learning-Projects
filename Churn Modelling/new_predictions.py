# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:25:09 2021

@author: clovi
"""

'''from sklearn.preprocessing import StandardScaler
ss = StandardScaler()'''

def NewPredictions(ann, ss, predict =''):
    
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
        
        
        new_predict = ann.predict(ss.transform(data_list))
        prob_predict = round((float(new_predict))*100,2)
        bool_predict = (bool(new_predict > 0.5))
        
        result = (prob_predict,bool_predict) 
        predictions.append(result)
        
        print('The probability of the customer to leave the bank is about', prob_predict, '%.')
        
        verif = str(input("Type 'exit' to exit the program or press anything to make another prediction. \n")).lower()
        if verif == 'exit':
            return predictions
