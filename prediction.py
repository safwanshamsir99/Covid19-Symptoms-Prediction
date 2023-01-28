# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:32:02 2023

@author: Acer
"""

import pickle

MODEL_PATH = 'best_model_covid.pkl'

def predict(temp):
    with open(MODEL_PATH,'rb') as file:
        model = pickle.load(file)
    return model.predict(temp)
