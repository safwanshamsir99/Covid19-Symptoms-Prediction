# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 19:20:54 2022

@author: Acer
"""

import streamlit as st
import os
import numpy as np
import pickle

MODEL_PATH = 'https://github.com/safwanshamsir99/Covid19-Symptoms-Prediction/blob/main/best_model_covid.pkl'

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)


#%% STREAMLIT
with st.form("Patient's Form"):
    st.title("Prediction of Covid-19 Based on Symptoms")
    st.video("https://www.youtube.com/watch?v=U8r3oTVMtQ0", format="video/mp4") 
    # credit video: "How Old Is Your Heart? Learn Your Heart Age!" By CDC YouTube channel
    st.header("Let's check if you have infected by Covid-19 virus!")
    st.write("0 = No,",
             "\n 1 = Yes.")
    breathing_problem = int(st.radio("Do you have breathing problem?", (0,1)))
    fever = int(st.radio("Do you experience fever?", (0,1)))
    dry_cough = int(st.radio("Do you get dry cough?", (0,1)))
    sore_throat = int(st.radio("Do you experience sore throat?", (0,1)))
    hyper_tension = int(st.radio("Do you encounter hyper tension for a few moment?", (0,1)))
    abroad_travel = int(st.radio("Do you travel abroad recently?", (0,1)))
    contact_with_covid_patient = int(st.radio("Do you meet anyone that is positive with Covid-19 recently?", (0,1)))
    attended_large_gathering = int(st.radio("Do you attend any gathering recently?", (0,1)))
    visited_public_exposed_places = int(st.radio("Do you visit any public exposed place?", (0,1)))
    family_working_in_public_exposed_places = int(st.radio("Do you have any family member that works in public exposed place?", (0,1)))
    

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Breathing problem:",breathing_problem,
                 "Fever:",fever,
                 "Dry cough:",dry_cough,
                 "Sore throat:",sore_throat,
                 "Hypertension:",hyper_tension,
                 "Abroad travel:",abroad_travel,
                 "Contact with Covid-19:",contact_with_covid_patient,
                 "Attending large gathering:",attended_large_gathering,
                 "Visit public place:",visited_public_exposed_places,
                 "Family working in public place:",family_working_in_public_exposed_places)
        temp = np.expand_dims([breathing_problem,fever,dry_cough,sore_throat,
                               hyper_tension,abroad_travel,contact_with_covid_patient,
                               attended_large_gathering,visited_public_exposed_places,
                               family_working_in_public_exposed_places], axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0:'Negative infection of Covid-19',
                        1:'Positive infection of Covid-19'}
        
        if outcome == 1:
            st.snow()
            st.markdown('**POSITIVE!** You are infected with Covid-19 virus!')
            st.write("Please follow any quarantine or isolation instructions, practice good hygiene and seek medical attention if your symptoms worsen!")
            st.image("https://www.aicb.org.my/images/announcement/covid-19-positive/covwtd_01.jpg")
            # Credit pic: aicb.org.my website
        else:
            st.balloons()
            st.write("Hooray, you are not infected by Covid-19 virus. Please practice the SOP guidelines to combat Covid-19 disease from spreading!")
            st.image("https://cdn.who.int/media/images/default-source/health-topics/coronavirus/myth-busters/infographic-covid-19-transmission-and-protections-final2.jpg?sfvrsn=7fc5264a_2")
            # Credit pic: cdn.who.int/graphics website
        
