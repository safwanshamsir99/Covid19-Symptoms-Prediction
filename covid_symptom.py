# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:02:46 2022

@author: Acer
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#%% FUNCTION
def plot_cat(df,cat_data):
    '''
    This function is to generate plots for categorical columns

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    cat_data : LIST
        categorical column inside the dataframe.

    Returns
    -------
    None.

    '''
    for cat in cat_data:
        plt.figure()
        sns.countplot(df[cat], hue=df['COVID-19'])
        plt.show()

def plot_target(df,target_column):
    '''
    This function is to generate plots for continuous columns

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    continuous_col : LIST
        continuous column inside the dataframe.

    Returns
    -------
    None.

    '''
    plt.figure()
    sns.countplot(target_column)
    plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,         
        Journal of the Korean Statistical Society 42 (2013): 323-328    
    """    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STATIC
#CSV_URL = os.path.join(os.getcwd(),'covid_symptom.csv')

#%% DATA LOADING
df = pd.read_csv('covid_symptom.csv')

#%% DATA INSPECTION
df.info() # all column is object datatype
df.isna().sum() # no null values
df.duplicated().sum()#4968 duplicated but cant be considered as duplicated since it is a categorical (yes/no) datatype
df.columns
df.describe(include='all')

#%% DATA VISUALIZATION
cat_column = df.loc[:,df.columns != 'COVID-19']
plot_cat(df, cat_column) 

plot_target(df, df['COVID-19']) # unbalance target column

#%% DATA PREPROCESSING
# Create a LabelEncoder instance
le = LabelEncoder()
cat_data = df.columns

# Fit the LabelEncoder to the categorical feature column in your DataFrame
for cat in cat_data:
  df[cat] = le.fit_transform(df[cat])

# Save the LabelEncoder instance to a file using pickle
LE_PATH = os.path.join(os.getcwd(),'le_covid.pkl')
with open(LE_PATH,'wb') as file:
  pickle.dump(le,file)
  
# Check correlation
for cat in cat_column:
    confussion_mat = pd.crosstab(df[cat],df['COVID-19']).to_numpy()
    print(cat + ':' + str(cramers_corrected_stat(confussion_mat)))
    
# Visualize the correlation
cor = df.corr()
cor.style.background_gradient(cmap='coolwarm',axis=None)

#%% FEATURES SELECTION & FORMATTING
delete_features = ['Running Nose','Asthma','Chronic Lung Disease','Headache',
                   'Heart Disease','Diabetes','Fatigue ','Gastrointestinal ',
                   'Wearing Masks','Sanitization from Market']
for delete in delete_features:
  df = df.drop(delete,axis=1)
  
# Replace whitespace in column names into underscore
df.columns = df.columns.str.replace(' ', '_')

# Change capital letter into lowercase letter
df.columns = df.columns.str.title().str.lower()

X = df.loc[:,df.columns != 'covid-19']
y = df.loc[:,'covid-19']

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=3)

#%% DATA MODELLING
models = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),
          KNeighborsClassifier(),SVC(),GradientBoostingClassifier()]

model_names=['Logistic Regression Classifier','Random Forest Classifier',
             'Decision Tree Classifier','KNeighbors Classifier',
             'Support Vector Classifier','Gradient Boosting Classifier']

# fitting the data
for ml in models:
    ml.fit(X_train,y_train)

ml_dict = {0:'Logistic Regression Classifier',
           1:'Random Forest Classifier',
           2:'Decision Tree Classifier',
           3:'KNeighbors Classifier',
           4:'Support Vector Classifier',
           5:'Gradient Boosting Classifier'}
best_accuracy = 0

# model evaluation
scores = []
for i, ml in enumerate(models):
  scores = [ml.score(X_test, y_test) for ml in models]
  if ml.score(X_test, y_test) > best_accuracy:
      best_accuracy = ml.score(X_test,y_test)
      best_result = ml
      best_model = ml_dict[i]
  result = pd.DataFrame({
      'Model':model_names,
      'Score':scores})
print(result)
       
print('The best machine learning model for this dataset will be {} with accuracy of {}'
      .format(best_model, best_accuracy))

#%% RETRAIN THE MODEL TO SAVE THE BEST MODEL
rf = RandomForestClassifier()
ml_model = rf.fit(X_train,y_train)

# saving the best model
MODEL_PATH = os.path.join(os.getcwd(),'best_model_covid.pkl')
with open(MODEL_PATH,'wb') as file:
  pickle.dump(ml_model,file)

#%% EVALUATION PART
y_true = y_test
y_pred = ml_model.predict(X_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print('Accuracy score: ' + str(accuracy_score(y_true, y_pred)))


