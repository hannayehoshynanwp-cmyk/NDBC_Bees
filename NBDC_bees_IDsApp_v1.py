import streamlit as st

from PIL import Image

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing import image

import plotly.express as px



def main():
  # Set up your web app

  st.title('NDBC Bees Identification with MobileNetV3')

  # Upload image file
  uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    st.success("Image uploaded successfully!")
  else:
    st.info("Please upload an image file.")
  
 

if __name__ == '__main__':

    main()









