# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:22:09 2021

@author: clovi
"""
import dropdown

def NewPredictions(ann, ss, work_dict, education_dict, marital_dict, occupation_dict, relationship_dict, race_dict, country_dict):
    
    predictions = []
    while True:
        data_list = []
        final_data = []
        gender_dict = {'male': 1, 'female': 0}
        
        workclass = str(dropdown.drop_workclass()).replace("-","").replace(" ","").lower()
        education = str(dropdown.drop_education()).replace("-","").replace(" ","").lower()
        marital = str(dropdown.drop_marital()).replace("-","").replace(" ","").lower()
        occupation = str(dropdown.drop_occupation()).replace("-","").replace(" ","").lower()
        relationship = str(dropdown.drop_relationship()).replace("-","").replace(" ","").lower()
        race = str(dropdown.drop_race()).replace("-","").replace(" ","").lower()
        gender = str(dropdown.drop_sex()).replace("-","").replace(" ","").lower()
        country = str(dropdown.drop_country()).replace("-","").replace(" ","").lower()
        age = dropdown.get_age()
        education_num = dropdown.get_ednum()
        capital_gain = dropdown.get_gain()
        capital_loss = dropdown.get_loss()
        hours = dropdown.get_hours()
        
        
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

