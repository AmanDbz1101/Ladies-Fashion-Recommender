import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing import image 
# from PIL import Image 
import numpy as np
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import streamlit as st 

import pandas as pd
from PIL import Image
import io 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
# with open('features.pkl','rb') as file:
#     feature_list = pickle.load(file)
# # file_names = pickle.load('file_names.pkl', 'rb')

# from datasets import load_dataset
# ds = load_dataset("yotam56/hugo_dresses_ds", split="train")


with open('features2.pkl','rb' ) as file:
    feature_list = pickle.load(file)
    
ds = pd.read_parquet('data/first.parquet')

st.title("Fashion Recommender System")

cv = CountVectorizer(max_features= 500, stop_words='english')

text = st.text_area("Enter your text", height=100)

if text:
    text_data = pd.Series(text)
    text_df = ds['text']

    new_df = pd.concat([text_data, text_df])

    vectors = cv.fit_transform(new_df).toarray()

    similarity = cosine_similarity(vectors)

    distances = similarity[0]
    recommended_list = sorted(list(enumerate(distances)),reverse=True, key = lambda x:x[1])[1:7]


    row1= st.columns(3)
    row2 = st.columns(3)
    i=0
    for col in row1+row2:
        tile = col.container(height=240)
        image_file = ds['image'].iloc[recommended_list[i][0]-1]  # Access the first image
        image_data =image_file['bytes']
        # Decode the image
        image = Image.open(io.BytesIO(image_data))
        tile.image(image)
        i+=1     

uploaded_file = st.file_uploader("Drop the above images here or insert your image")

if uploaded_file is not None:
    
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False 
    model = Sequential([model, GlobalMaxPool2D()])
    
    
    img = uploaded_file
    image = load_img(img, target_size=(224, 224))
    image_array = img_to_array(image)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    processed_input = preprocess_input(expanded_image_array)
    extracted_feature = model.predict(processed_input).flatten()
    normalized_extracted_feature = extracted_feature/norm(extracted_feature)


    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([normalized_extracted_feature])



    list = indices.flatten().tolist()

    row1= st.columns(3)
    row2 = st.columns(3)
    i=0
    for col in row1+row2:
        tile = col.container(height=240)
        image_file = ds['image'].iloc[list[i]]  # Access the first image
        image_data =image_file['bytes']
        # Decode the image
        image = Image.open(io.BytesIO(image_data))
        tile.image(image)
        i+=1