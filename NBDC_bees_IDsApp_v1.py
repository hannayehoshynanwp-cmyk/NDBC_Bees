import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import keras
import plotly.express as px



def main():
  #Img preprocessing
  def preprocess_img(beeImgFile, IMG_SIZE=416):
    #st.info('Preprocessing...')
    rawImg = keras.utils.load_img(beeImgFile, target_size=(IMG_SIZE,IMG_SIZE))
    imgArr = keras.utils.array_to_img(rawImg)
    #rawImg=image.load_img(beeImgFile, target_size=(IMG_SIZE,IMG_SIZE))
    #imgArr=image.img_to_array(rawImg)
    imgArrExp = np.expand_dims(imgArr, 0)
    beeImg=tf.keras.applications.mobilenet_v3.preprocess_input(imgArrExp)
    return beeImg
    
  def load_model(model_file):
    model = joblib.load(model_file)
    return model
    
  # Set up your web app

  st.title('NDBC Bees Identification with MobileNetV3')

  # Upload image file
  uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    st.success("Image uploaded successfully!")
    
    # Button to make a prediction
    if st.button('Predict'):
      try:
        beeImgFile=preprocess_img(uploaded_file)
        st.write("")
        #st.write("Identification...")
        with st.spinner(text='Identification in progress... Please wait.'):
          model = load_model('model1_v3_NDBC_Bees_8_8_25.pkl')
          labels=joblib.load('model1_v3_NDBC_Bees_8_8_25_labels.pkl')
          testImgPreds=model.predict(beeImgFile)
          st.success(f"Top class is: {labels[testImgPreds[0].argmax()]}")
      except Exception as e:
          st.error(f"Error in prediction: {e}")

  
  else:
    st.info("Please upload an image file.")
    
  
 

if __name__ == '__main__':

    main()












