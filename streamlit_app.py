import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Recycle Analyser")

st.write("## To identify whether your item can be recycled")
st.write(
    ":dog: Item :grin:"
)
st.sidebar.write("## Upload :gear:")

# Global Parameters
imageShape = 224

recycleSet = pd.read_csv('./imgModel.csv')
recycleSet = recycleSet[recycleSet['Include'] == 'Yes']

# Data Items
dataClasses = np.array(recycleSet['Item'])
recycleClasses = np.array(recycleSet['Recycle'])

# Model Parameters
selectModel = tf.keras.models.load_model('./recycleModel.h5')

def analyseItem(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)
   
    img = tf.keras.preprocessing.image.load_img(upload, target_size=(imageShape, imageShape))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    modelPrediction = selectModel.predict(img_array)
    predictedId = np.argmax(modelPrediction, axis=-1)
    predictedLabel = dataClasses[predictedId]
    
    col2.write(predictedLabel[0].title() + ' (Recyclable: ' + recycleClasses[predictedId[0]] + ') :wrench:')

# Streamlit Visible
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    analyseItem(upload=my_upload)
else:
    analyseItem("./1.JPEG")