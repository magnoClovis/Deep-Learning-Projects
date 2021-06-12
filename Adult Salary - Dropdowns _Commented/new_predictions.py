# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:22:09 2021

@author: clovi
"""

'''
Now, for making new predictions, it is going to be used the dictionaries with the
encodings that has been created later on, and also it is needed to be passed to
the NewPredictions function the StandardScaler used, so then, the program is able
to standarize the new date using the same pattern it has been used before.

It is being used here functions from "dropdown.py", this file is responsible for 
generating the interaction windows and boxes for getting the inputs.
'''
import dropdown

def NewPredictions(ann, ss, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict):
    
    predictions = [] # This empty list will store the results obtained at the end
    while True:
        data_list = [] # Here, the data that is going to be encodade with the dictionary will be stored
        final_data = [] # Here it will be stored all the new data, it is this list that is going to be used in the prediction
        gender_dict = {'male': 1, 'female': 0} 
        
        '''
        The lines below get the input from the user and filter it in the same 
        standar use before to filter the dictionaries keys. Therefore it is 
        possible to match the keys with no problems
        '''
        workclass = str(dropdown.drop_workclass()).replace("-","").replace(" ","").replace("Other","?").lower()
        education = str(dropdown.drop_education()).replace("-","").replace(" ","").lower()
        marital = str(dropdown.drop_marital()).replace("-","").replace(" ","").lower()
        occupation = str(dropdown.drop_occupation()).replace("-","").replace(" ","").replace("Other","?").lower()
        relationship = str(dropdown.drop_relationship()).replace("-","").replace(" ","").lower()
        race = str(dropdown.drop_race()).replace("-","").replace(" ","").lower()
        gender = str(dropdown.drop_sex()).replace("-","").replace(" ","").lower()
        country = str(dropdown.drop_country()).replace("-","").replace(" ","").replace("Other","?").lower()
        
        '''
        These last inputs are note encodable data, then there is no need to 
        filter or standarize it as before
        '''
        age = dropdown.get_age()
        education_num = dropdown.get_ednum()
        capital_gain = dropdown.get_gain()
        capital_loss = dropdown.get_loss()
        hours = dropdown.get_hours()
        
        '''
        The dictionaries is now being used to get the code of each of the inputs
        above
        '''
        workclass_lst = work_dict[workclass]
        education_lst = education_dict[education]
        marital_lst = marital_dict[marital]
        occupation_lst = occupation_dict[occupation]
        relationship_lst = relationship_dict[relationship]
        race_lst = race_dict[race]
        country_lst = country_dict[country]
        gender_int = gender_dict[gender]


        '''
        Then now it is going to be stored in another list the values of each
        list created to each variables. They will, together, compose an entire
        list with one line an several columns, in other words, it is not going
        to be a list of list, instead, it is going to be only one list with all
        the values in sequence. That is what is being done with the for loops.
        
        
        Order of the data: Workclass, Education, Marital Status, Occupation
        Relationship, Race, Native Country, Age, Education Number, Gender,
        Capital Gain, Capital Loss, Hours per week
        '''
        att_list = [workclass_lst, education_lst, marital_lst, occupation_lst,
                    relationship_lst, race_lst, country_lst]
        
        for j in range(len(att_list)):
            for i in range(len(att_list[j])):
                data_list.append(att_list[j][i])
                
        att_num = [age, education_num, gender_int, capital_gain, capital_loss, hours]
        
        for j in range(len(att_num)):
            data_list.append(att_num[j])
                
        final_data.append(data_list)
    
        new_predict = ann.predict(ss.transform(final_data))
        prob_predict = round((float(new_predict))*100,2)
        bool_predict = (bool(new_predict > 0.5))
        

        
        result = (prob_predict,bool_predict) 
        predictions.append(result)
        
        
        verif = dropdown.drop_exit(bool_predict, prob_predict)
        if verif == 'Yes':
            return predictions

