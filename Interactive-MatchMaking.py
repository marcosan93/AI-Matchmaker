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
with open("clustered_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)
    
    
    
    
## Fitting the Vectorizer and Scaler to the Data
X = df.drop(["Cluster #"], 1)

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
    Scales and vectorizes the new data by using the previously fitted scaler and vectorizer then 
    returns that new DF
    """
    # Vectorizing the new data
    vect_new_prof = vectorizer.transform(new_df['Bios'])

    # Quick DF of the vectorized words
    new_vect_w = pd.DataFrame(vect_new_prof.toarray(), columns=vectorizer.get_feature_names(), index=new_profile.index)

    # Concatenating the DFs for the new profile data
    new_vect_prof = pd.concat([new_profile, new_vect_w], 1).drop('Bios', 1)

    # Scaling the new profile data
    new_vect_prof = pd.DataFrame(scaler.transform(new_vect_prof), columns=new_vect_prof.columns, index=new_vect_prof.index)
    
    return new_vect_prof


def top_ten(cluster, new_profile):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = cluster_df[cluster_df['Cluster #']==designated_cluster[0]]
    
    # Appending the new profile data
    des_cluster = des_cluster.append(new_profile, sort=False)

    # Fitting the vectorizer to the Bios
    cluster_x = vectorizer.fit_transform(des_cluster['Bios'])

    # Creating a new DF that contains the vectorized words
    cluster_v = pd.DataFrame(cluster_x.toarray(), index=des_cluster.index, columns=vectorizer.get_feature_names())

    # Joining the Vectorized DF to the previous DF and dropping columns
    des_cluster = des_cluster.join(cluster_v).drop(['Bios', 'Cluster #'], axis=1)
    
    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corr()

    # Finding the Top 10 similar or correlated users to the new user
    user_n = new_profile.index[0]

    # Creating a DF with the Top 10 most similar profiles
    top_10_sim = corr[[user_n]].sort_values(by=[user_n],axis=0, ascending=False)[1:11]
    
    return df.drop('Cluster #', 1).loc[top_10_sim.index]




