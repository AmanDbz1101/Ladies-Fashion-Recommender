from sklearn.feature_extraction.text import CountVectorizer
import io
from PIL import Image
import pandas as pd
import streamlit as st 
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_parquet('data/first.parquet')

cv = CountVectorizer(max_features= 500, stop_words='english')

st.title("Upper wear on Text")

text = st.text_area("Enter your text", height=100)




if text:
    text_data = pd.Series(text)
    text_df = df['text']

    new_df = pd.concat([text_data, text_df])

    vectors = cv.fit_transform(new_df).toarray()

    similarity = cosine_similarity(vectors)

    distances = similarity[0]
    recommended_list = sorted(list(enumerate(distances)),reverse=True, key = lambda x:x[1])[1:7]

    # for i in recommended_list:
    #     image_file = df['conditioning_image'].iloc[i[0]-1]  # Access the first image
    #     image_data =image_file['bytes']
    #     # Decode the image
    #     image = Image.open(io.BytesIO(image_data))


    row1= st.columns(3)
    row2 = st.columns(3)
    i=0
    for col in row1+row2:
        tile = col.container(height=400)
        image_file = df['image'].iloc[recommended_list[i][0]-1]  # Access the first image
        image_data =image_file['bytes']
        # Decode the image
        image = Image.open(io.BytesIO(image_data))
        tile.image(image)
        i+=1
