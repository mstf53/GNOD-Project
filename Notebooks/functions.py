
import pandas as pd
import numpy as np
import time
import spotipy
import json
import sys
from config import *
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

# connect to spotify database
def connect_spotify():
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= Client_ID,
                                                           client_secret= Client_Secret)) #connects to spotify api
    return sp


# Search a song in Spotify and return a df with audio features
def search_song(sp, title, artist):
    try:
        #sp = connect_spotify()
        song = sp.search(q=f'track:{title} artist:{artist}', limit=1)
        #print(song)
        song_id = song["tracks"]["items"][0]["id"]
        song_with_feature = sp.audio_features(song_id)
        column = list(song_with_feature[0].keys())
        values = [list(song_with_feature[0].values())]
        df_new_song = pd.DataFrame(data = song_with_feature, columns = column)
        return df_new_song
    except:
        print('Unfortunately this song is not part of our catalogue, maybe you would like to try something more common?')
        return None
    

# main function
def song_recommend():
    
    answer = "Yes"
    while (answer == "Yes"):
        
        user_title = input("Type a song title: ")
        user_artist = input("Type an artist: ")


        #search_song

        #get audio features

        sp = connect_spotify()
        df_new_song = search_song(sp, user_title, user_artist)
        
        if ( type(df_new_song) == pd.DataFrame ):
            numericals = df_new_song.select_dtypes(include=np.number)
            numericals = numericals.drop(['time_signature'], axis=1)

            # Standardizing the df that we got from search_song
            scaler = load("../scaler/scaler.pickle")
            cols = load("cols.pickle")
            X_scaled = scaler.transform(numericals[cols])
            X_scaled_df = pd.DataFrame(X_scaled)

            # applying model(kmeans with k=14 )
            model = load('../models/kmeans_14.pickle')
            clusters = model.predict(X_scaled_df)

            # create new df with cluster that gives the model
            df_joined = pd.read_csv("../data/clean/df_joined.csv")
            df_last = df_joined[df_joined['cluster'] == list(clusters)[0]]

            # random choice of a song
            recommendation = df_last.sample()

            #print(list(recommendation.columns).index("id"))
            print('Maybe you would like to listen to your NEW favourite song, which is:') 
            #{recommendation.iloc[0,1]} by {recommendation.iloc[0,2]}')
            print('')
            print('Song title:', recommendation.iloc[0,0]) 
            print('by:',recommendation.iloc[0,1])
            print('')  
            print("Link: ")
            print("https://open.spotify.com/track/"+recommendation.iloc[0,14])
            print("_________________________")
            user = input("Would you like another song recommendation? [Yes/No]")
            print('')    
            #return recommendation
            while ( user not in ["Yes", "No"] ):
                print("Are you blind? I said Yes or No!")
                user = input("Would you like another song recommendation? [Yes/No]")
            answer = user

        
def load(filename = "filename.pickle"): 
    try: 
        with open(filename, "rb") as file: 
            return pickle.load(file) 
    except FileNotFoundError: 
        print("File not found!") 
