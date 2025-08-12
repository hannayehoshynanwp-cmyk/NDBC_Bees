import streamlit as st
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing import image




def main():
  # Set up your web app

  st.title('NDBC Bees Identification with MobileNetV3')
  st.write('Enter the filename to predict its Bee species')
  # User input
  beeImgFile=st.text_area('', height=200)
  # Button to make a prediction
  if st.button('Predict'):
        try:
          beeImgFile=preprocess_img(beeImgFile)
          model = load_model('model1_v3_NDBC_Bees_8_8_25.pkl')
          #Load labels
          labels=pickle.load(open('model1_v3_NDBC_Bees_8_8_25_labels.pkl', 'rb'))
          # Make a prediction
          testImgPreds=model.predict(preprocess_img(beeImgFile))
          # Display the result
          st.write(f"Top class is: {labels[testImgPreds[0].argmax()]}")
        except Exception as e:
          st.error(f"Error in prediction: {e}")



  # Load your model (with caching to improve performance)
  @st.cache_resource()
  def load_model(model_file):
      model = pickle.load(open(model_file, 'rb'))
      return model

  #Img preprocessing
  def preprocess_img(beeImgFile, IMG_SIZE=416):
    rawImg=image.load_img(beeImgFile, target_size=(IMG_SIZE,IMG_SIZE))
    imgArr=image.img_to_array(rawImg)
    imgArrExp = np.expand_dims(imgArr, 0)
    beeImg=tf.keras.applications.mobilenet_v3.preprocess_input(imgArrExp)
    return beeImg

if __name__ == '__main__':
    main()


  

  



  

