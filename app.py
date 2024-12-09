import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing import image 
from PIL import Image 
import numpy as np
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import streamlit as st 

with open('features.pkl','rb') as file:
    feature_list = pickle.load(file)
# file_names = pickle.load('file_names.pkl', 'rb')




st.title("Fashion Recommender System")

uploaded_file = st.file_uploader("Choose a file")

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

    from datasets import load_dataset
    ds = load_dataset("yotam56/hugo_dresses_ds", split="train")

    list = indices.flatten().tolist()

    row1= st.columns(3)
    row2 = st.columns(3)
    i=0
    for col in row1+row2:
        tile = col.container(height=400)
        tile.image(ds["image"][list[i]])
        i+=1
                

