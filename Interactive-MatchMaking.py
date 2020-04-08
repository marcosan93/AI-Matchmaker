# Library Imports
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import _pickle as pickle
from random import sample

# Loading the Profiles
with open("clustered_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)
    
# Loading the Classification Model
model = load("clf_model.joblib")
    
      
## Fitting the Vectorizer and Scaler to the Data
X = df.drop(["Cluster #"], 1)

# Instantiating the Vectorizer
vectorizer = CountVectorizer()

# Fitting the vectorizer to the Bios
x = vectorizer.fit_transform(X['Bios'].values.astype('U'))

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
    vect_new_prof = vectorizer.transform(new_df['Bios'].values.astype('U'))

    # Quick DF of the vectorized words
    new_vect_w = pd.DataFrame(vect_new_prof.toarray(), columns=vectorizer.get_feature_names(), index=new_df.index)

    # Concatenating the DFs for the new profile data
    new_vect_prof = pd.concat([new_df, new_vect_w], 1).drop('Bios', 1)

    # Scaling the new profile data
    new_vect_prof = pd.DataFrame(scaler.transform(new_vect_prof), columns=new_vect_prof.columns, index=new_vect_prof.index)

    return new_vect_prof


def top_ten(cluster, new_profile):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = df[df['Cluster #']==cluster[0]]
    
    # Appending the new profile data
    des_cluster = des_cluster.append(new_profile, sort=False)

    # Fitting the vectorizer to the Bios
    cluster_x = vectorizer.fit_transform(des_cluster['Bios'].values.astype('U'))

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
    
    # The Top Profiles
    top_10 = df.drop('Cluster #', 1).loc[top_10_sim.index]
    
    # Converting the floats to ints
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]].astype(int)
    
    return top_10


## Interactive Section
st.title("AI-MatchMaker")

st.markdown("Finding your Top 10 Profiles")

# Instantiating a new DF row to append later
new_profile = pd.DataFrame(columns=df.drop('Cluster #', 1).columns, index=[df.index[-1]+1])

# Example Bios for the user
st.write("-"*100)
st.text("Example Bios:")
for i in sample(list(df.index), 3):
    st.text(df['Bios'].loc[i])
st.write("-"*100)
    
# Asking for new profile data
new_profile['Bios'] = st.text_input("Enter a Bio for yourself: ")

random_vals = st.checkbox("Check here if you would like random values for yourself")

if random_vals:
    # Adding random values for new data
    for i in new_profile.columns[1:]:
        new_profile[i] = np.random.randint(0,10,1)

else:
    for i in new_profile.columns[1:]:
        new_profile[i] = st.selectbox(f"Select value for {i}", [i for i in range(10)])

button = st.button("Click to find your Top 10!")

if button:    
    with st.spinner('Finding your Top 10 Matches'):
        # Formatting the New Data
        new_df = prep_new_data(new_profile)
        
        # Predicting/Classifying the new data
        cluster = model.predict(new_df)

        # Finding the top 10 related profiles
        top_10_df = top_ten(cluster, new_profile)

        # Displaying the Top 10 similar profiles
        st.dataframe(top_10_df, width=10000)

    # Success message   
    st.success("Found your Top 10!")    
    st.balloons()

    

