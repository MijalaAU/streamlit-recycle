import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Recycle Analyser")

st.write("## To identify whether your item can be recycled")
st.write("### :cat: Michael Lamb - 220545523")
st.write('### :grin: Deakin University SIT744 - 23T1 | Assignment 2')
st.sidebar.write("## :gear: Upload")

# Global Parameters
imageShape = 224

recycleSet = pd.read_csv('./imgModel.csv')
recycleSet = recycleSet[recycleSet['Include'] == 'Yes']

# Data Items
dataClasses = np.array(recycleSet['Item'])
recycleClasses = np.array(recycleSet['Recycle'])

# Model Parameters
#selectModel = tf.keras.models.load_model('./recycleModel.h5')

def analyseItem(upload):
    image = Image.open(upload)
    st.write("Original Sample :camera:")
    st.image(image)
   
    #img = tf.keras.preprocessing.image.load_img(upload, target_size=(imageShape, imageShape))
    #img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = tf.expand_dims(img_array, 0)

    #modelPrediction = selectModel.predict(img_array)
    #predictedId = np.argmax(modelPrediction, axis=-1)
    #predictedLabel = dataClasses[predictedId]
    
    st.write("### :wrench: Prediction: " + 'Leftovers' + ' (Recyclable: ' + 'No' + ')') # predictedLabel[0].title(), recycleClasses[predictedId[0]]

# Streamlit Visible
col1 = st.columns(1)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    analyseItem(upload=my_upload)
else:
    analyseItem("./1.JPEG")