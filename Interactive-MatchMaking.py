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