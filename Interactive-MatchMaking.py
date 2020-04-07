# Library Imports
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import _pickle as pickle

# Loading the Profiles
with open("profiles.pkl",'rb') as fp:
    df = pickle.load(fp)
    
    
## Fitting the Vectorizer and Scaler to the Data

# Instantiating the Vectorizer
vectorizer = CountVectorizer()

# Fitting the vectorizer to the Bios
x = vectorizer.fit_transform(X['Bios'])

# Creating a new DF that contains the vectorized words
df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

# Concating the words DF with the original DF
X = pd.concat([X, df_wrds], axis=1)

# Dropping the Bios because it is no longer needed in place of vectorization
X.drop(['Bios'], axis=1, inplace=True)

# Instantiating the Scaler
scaler = MinMaxScaler()

scaler.fit(X)


## Helper Functions

def prep_new_data(new_df):
    """
    Scales and vectorizes the new data by using the previously fitted scaler and vectorizer
    """
    # Vectorizing the new data
    vect_new_prof = vectorizer.transform(new_df['Bios'])

    # Quick DF of the vectorized words
    new_vect_w = pd.DataFrame(vect_new_prof.toarray(), columns=vectorizer.get_feature_names(), index=new_profile.index)

    # Concatenating the DFs for the new profile data
    new_vect_prof = pd.concat([new_profile, new_vect_w], 1).drop('Bios', 1)

    # Scaling the new profile data
    new_vect_prof = pd.DataFrame(scaler.transform(new_vect_prof), columns=new_vect_prof.columns, index=new_vect_prof.index)


