# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:32:02 2023

@author: Acer
"""

import joblib

def predict(temp):
    model = joblib.load('best_model_covid.sav')
    return model.predict(temp)
