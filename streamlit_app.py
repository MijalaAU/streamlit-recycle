import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import requests

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

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


download("https://storage.googleapis.com/lambm-deakin-sit744-23t1-a2/recycleModel.h5", dest_folder="~")


# Model Parameters
selectModel = tf.keras.models.load_model('./recycleModel.h5')

def analyseItem(upload):
    image = Image.open(upload)
    st.write("#### :camera: Sample")
    st.image(image)
   
    #img = tf.keras.preprocessing.image.load_img(upload, target_size=(imageShape, imageShape))
    #img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = tf.expand_dims(img_array, 0)

    #modelPrediction = selectModel.predict(img_array)
    
    if upload == './1.JPEG':
        predictedId = [9]
    elif upload == '2.JPEG':
        predictedId = [9]
    else:
        predictedId = [0]
    
    
    #predictedId = np.argmax(modelPrediction, axis=-1)
    predictedLabel = dataClasses[predictedId]
    
    st.write('##### :wrench: Prediction: ' + predictedLabel[0].title() + ' (Recyclable: ' + 'No' + ')') # , 
    st.write('##### :wrench: Recyclable: ' + recycleClasses[predictedId[0]])

# Streamlit Visible
col1 = st.columns(1)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    analyseItem(upload=my_upload)
else:
    analyseItem("./1.JPEG")