# import modul yang akan digunakan
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

st.cache_data ()
def load_data():

    #load dataset
    df = pd.read_csv('gender_classification_v7.csv')
  
    x = df[['long_hair', 'forehead_width_cm', 'forehead_height_cm', 'nose_wide',	'nose_long', 'lips_thin', 'distance_nose_to_lip_long']]
    y = df['gender']

    return df, x, y

st.cache_data ()
def train_model(x,y):
    model=KNeighborsClassifier(n_neighbors=3)
    
    model.fit(x,y)

    score = model.score(x,y)

    return model, score

def predict(x,y, features):
    model, score = train_model(x,y)

    prediction = model.predict(np.array(features).reshape(1,-1))

    return prediction, score