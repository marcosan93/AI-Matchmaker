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
from PIL import Image
from scipy.stats import halfnorm

# Loading the Profiles
with open("refined_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)
    
# Loading the Classification Model
model = load("refined_model.joblib")


## Helper Functions

def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x
 
    
def prep_data(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    column_name = columns[0]
    
    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
        return df, input_df
    
    # Encoding columns with respective values
    if column_name in ['Religion', 'Politics']:
        
        # Getting labels for the original df
        df[column_name.lower()] = df[column_name].cat.codes
        
        # Dictionary for the codes
        d = dict(enumerate(df[column_name].cat.categories))
        
        # Getting labels for the input_df
        input_df[column_name.lower()] = input_df[column_name].map(d)
        
        # Dropping the column names
        input_df = input_df.drop(column_name, 1)
        
        df = df.drop(column_name, 1)
        
        return vectorization(df, df.columns, input_df)
    
    # Vectorizing the other columns
    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()
        
        # Fitting the vectorizer to the columns
        x = vectorizer.fit_transform(df[column_name])
        
        y = vectorizer.transform(input_df[column_name])

        # Creating a new DF that contains the vectorized words
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
        
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names())

        # Concating the words DF with the original DF
        new_df = pd.concat([df, df_wrds], axis=1)
        
        y_df = pd.concat([input_df, y_wrds], 1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)
        
        y_df = y_df.drop(column_name, 1)
        
        return vectorization(new_df, new_df.columns, y_df) 

    
def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()
    
    scaler.fit(df)
    
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
    
    return input_vect
    


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


def example_bios():
    """
    Creates a list of randome example bios from the original dataset
    """
    # Example Bios for the user
    st.write("-"*100)
    st.text("Some example Bios:\n(Try to follow the same format)")
    for i in sample(list(df.index), 3):
        st.text(df['Bios'].loc[i])
    st.write("-"*100)

## Creating a List for each Category

# Probability dictionary
p = {}

# Movie Genres
movies = ['Adventure',
          'Action',
          'Drama',
          'Comedy',
          'Thriller',
          'Horror',
          'RomCom',
          'Musical',
          'Documentary']

p['Movies'] = [0.28,
               0.21,
               0.16,
               0.14,
               0.09,
               0.06,
               0.04,
               0.01, 
               0.01]

# TV Genres
tv = ['Comedy',
      'Drama',
      'Action/Adventure',
      'Suspense/Thriller',
      'Documentaries',
      'Crime/Mystery',
      'News',
      'SciFi',
      'History']

p['TV'] = [0.30,
           0.23,
           0.12,
           0.12,
           0.09,
           0.08,
           0.03,
           0.02,
           0.01]

# Religions (could potentially create a spectrum)
religion = ['Catholic',
            'Christian',
            'Jewish',
            'Muslim',
            'Hindu',
            'Buddhist',
            'Spiritual',
            'Other',
            'Agnostic',
            'Atheist']

p['Religion'] = [0.16,
                 0.16,
                 0.01,
                 0.19,
                 0.11,
                 0.05,
                 0.10,
                 0.09,
                 0.07,
                 0.06]

# Music
music = ['Rock',
         'HipHop',
         'Pop',
         'Country',
         'Latin',
         'EDM',
         'Gospel',
         'Jazz',
         'Classical']

p['Music'] = [0.30,
              0.23,
              0.20,
              0.10,
              0.06,
              0.04,
              0.03,
              0.02,
              0.02]

# Sports
sports = ['Football',
          'Baseball',
          'Basketball',
          'Hockey',
          'Soccer',
          'Other']

p['Sports'] = [0.34,
               0.30,
               0.16, 
               0.13,
               0.04,
               0.03]

# Politics (could also put on a spectrum)
politics = ['Liberal',
            'Progressive',
            'Centrist',
            'Moderate',
            'Conservative']

p['Politics'] = [0.26,
                 0.11,
                 0.11,
                 0.15,
                 0.37]

# Social Media
social = ['Facebook',
          'Youtube',
          'Twitter',
          'Reddit',
          'Instagram',
          'Pinterest',
          'LinkedIn',
          'SnapChat',
          'TikTok']

p['Social Media'] = [0.36,
                     0.27,
                     0.11,
                     0.09,
                     0.05,
                     0.03,
                     0.03,
                     0.03,
                     0.03]

age = None

# Lists of Names and the list of the lists
categories = [movies, religion, music, politics, social, sports, age]

names = ['Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Age']

combined = dict(zip(names, categories))
    
    
## Interactive Section

# Creating the Titles and Image
st.title("AI-MatchMaker")

st.header("Finding a Date with Artificial Intelligence")
st.write("Using Machine Learning to Find the Top Dating Profiles for you")

image = Image.open('robot_matchmaker.jpg')

st.image(image, use_column_width=True)

# Instantiating a new DF row to classify later
new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

# Asking for new profile data
new_profile['Bios'] = st.text_input("Enter a Bio for yourself: ")

# Printing out some example bios for the user        
example_bios()

# Checking if the user wants random bios instead
random_vals = st.checkbox("Check here if you would like random values for yourself instead")

# Entering values for the user
if random_vals:
    # Adding random values for new data
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = np.random.choice(combined[i], 1, p=p[i])
            
        elif i == 'Age':
            new_profile[i] = halfnorm.rvs(loc=18,scale=8, size=1).astype(int)
            
        else:
            new_profile[name] = list(np.random.choice(combined[i], size=1, p=p[name]))
            
            new_profile[name] = new_profile[name].apply(lambda x: list(set(x[0].tolist())))

else:
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])
            
        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)
            
        else:
            new_profile[name] = st.multiselect(f"What is your preferred choice for {i}?\n(Pick up to 3)", combined[i])
            
            new_profile[name] = new_profile[name].apply(lambda x: list(set(x[0].tolist())))
            
            
# Looping through the columns and applying the string_convert() function
for col in df.columns:
    df[col] = df[col].apply(string_convert)
    
    new_profile[col] = new_profile[col].apply(string_convert)
            

# Displaying the User's Profile        
st.table(new_profile)

# Push to start the matchmaking process
button = st.button("Click to find your Top 10!")

if button:    
    with st.spinner('Finding your Top 10 Matches...'):
        # Formatting the New Data
        new_df = prep_new_data(new_profile)
        
        # Predicting/Classifying the new data
        cluster = model.predict(new_df)

        # Finding the top 10 related profiles
        top_10_df = top_ten(cluster, new_profile)
        
        # Success message   
        st.success("Found your Top 10 Most Similar Profiles!")    
        st.balloons()

        # Displaying the Top 10 similar profiles
        st.table(top_10_df)

        

    

