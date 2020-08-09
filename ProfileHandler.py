import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler


# Creating the class object
class CreateProfile:
    
    def __init__(self, dataset=None, profile=None):
        """
        Using a given DF of profiles, creates a new profile based on information from that given DF
        
        If profile already given, allows formatting of that profile
        """
        
        # Checking if we have DFs in our arguments
        
        # Initializing instances of the smaller profile DF and the larger DF
        if type(dataset) != pd.core.frame.DataFrame:
            
            self.dataset = pd.DataFrame()
            
        else:
            self.dataset = dataset
                
        # Handling the profile
        if type(profile) != pd.core.frame.DataFrame:
            
            # Initializing a new DF for the new profile with a new index or user number
            self.profile = pd.DataFrame(index=[dataset.index[-1] + 1])
            
        else:
            # Using the given profile
            self.profile = profile
            
        # Vectorized version of the profile, will be N/A until the vect_text() method is used
        self.vectorized_text = "Use vect_text() to return a vectorized DF"
        
        # Scaled version of the profile, will be N/A until the scale_profile() method is used
        self.scaled_profile = "Use scale_profile() to return a scaled DF"
        
        # Formatted version of the profile, which contains the scaled and vectorized data of the profile
        self.formatted_profile = "Use format_profile() to return a both scaled and vectorized DF"
        
        # A combined DF containing both the original data and the new profile, will be N/A until add_profile_to_dataset() is used
        self.combined_df = "Use add_profile_to_dataset() to return the combined DF"
    
        
    def enter_info(self, random_info=True):
        """
        Enter information for the new profile either by user text input
        Or through random information from the larger dataset
        """
        
        if self.profile.empty:
            
            # Iterating through the columns of the larger profile in order to add new data to the smaller profile
            for i in self.dataset.columns:
                
                if random_info:
                    # Entering random information originally from profiles from the bigger profile
                    self.profile[i] = self.dataset[i].sample(1).to_numpy()
                
                else:
                    # Will need type checking
                    self.profile[i] = input(f"Enter info for {i}")
                    
            return self.profile
        
        else:
            
            # If there is already data in the profile
            return "Data already contained in the profile"
        
    
    def add_profile_to_dataset(self):
        """
        Appends the new profile to the dataset to return a new larger dataset containing the brand new profile
        
        Only will use the original format of the DF, no vectorized or scaled DFs
        """
        
        dataset_feats = self.dataset.columns
        
        profile_feats = self.profile.columns
                
        # Check to see if the profile profile contains the same features as the larger profile
        if dataset_feats.all()==profile_feats.all():
            
            # Appending the profile the larger profile
            self.combined_df = self.dataset.append(self.profile)
            
            return self.combined_df
        
        else:
            
            # If profile features/columns don't line up with the dataset's
            return "Profile features do not match larger dataset"
        
        
    def vect_text(self):
        """
        Given new profile data
        
        Replaces the text in the profile with a vectorized array of numbers.
        """
        
        # Finding all the text in the profile
        text = self.profile['Bios']
        
        # Instantiating the vectorizer
        vectorizer = CountVectorizer()
        
        # Fitting and transforming the text
        vect_words = vectorizer.fit_transform(text)
        
        # Converting the vectorized words into a DF
        self.vectorized_text = pd.DataFrame(vect_words.toarray(),
                                            index=self.profile.index,
                                            columns=vectorizer.get_feature_names())
        
        return self.vectorized_text
    
    
    def scale_profile(self, exclude=['Bios']):
        """
        Given a profile with information included
        
        Scales necessary features from the profile DF from 0 to 1 in relation the overall larger DF
        
        Does not scale features in the exclude list
        """
        
        # Instantiating the scaler we will use
        scaler = MinMaxScaler()
        
        # Creating a new DF for the scaled profile
        self.scaled_profile = pd.DataFrame(index=self.profile.index)
        
        # Iterating only through the necessary columns
        for col in self.dataset.columns:
            
            # Skipping columns we don't want to scale (i.e. text columns)
            if col in exclude:
                pass
            
            else:
                # Fitting the scaler to the larger DF
                scaler.fit(self.dataset[[col]])
                
                # Transforming the values based on the larger DF
                self.scaled_profile[col] = scaler.transform(self.profile[[col]])
                
        # Returning the final scaled profile
        return self.scaled_profile
                
        
    def format_profile(self):
        """
        Uses both scaling and vectorizing to format the profile DF
        """
        
        try:
            # If the attributes have already been instantiated by the methods before
            self.formatted_profile = pd.concat([self.scaled_profile, self.vectorized_text], axis=1)
            
        except:
            
            # If not, run the methods here
            self.formatted_profile = pd.concat([scale_profile(), vect_text()], axis=1)
        
        # Return the formatted profile DF
        return self.formatted_profile


