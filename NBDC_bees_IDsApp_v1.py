import streamlit as st
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf

from PIL import Image, ImageDraw

from tensorflow.keras.preprocessing import image

import plotly.express as px




def main():
  # Set up your web app

  st.title('NDBC Bees Identification with MobileNetV3')
  st.write('Enter the filename to predict its Bee species')
  # User input
  beeImgFile=st.text_area('', height=200)
  # Button to make a prediction
  if st.button('Predict'):
        try:
          uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
          if uploaded_file is not None:
            display_img(uploaded_file)
            beeImgFile=preprocess_img(beeImgFile)

            st.write("")
            st.write("Identification...")

            model = load_model('model1_v3_NDBC_Bees_8_8_25.pkl')
            #Load labels
            labels=pickle.load(open('model1_v3_NDBC_Bees_8_8_25_labels.pkl', 'rb'))
            # Make a prediction
            testImgPreds=model.predict(preprocess_img(beeImgFile))
            # Display the result
            st.success(f"Top class is: {labels[testImgPreds[0].argmax()]}")
            # Visualization
            display_predictions(testImgPreds,labels)

        except Exception as e:
          st.error(f"Error in prediction: {e}")
            # Load your model (with caching to improve performance)

  @st.cache_resource()
  def load_model(model_file):
      model = pickle.load(open(model_file, 'rb'))
      return model

  def display_img(imgFile):
    image = Image.open(imgFile)
    st.image(image, caption='Uploaded Image of Bee', use_column_width=True)

  def display_predictions(preds, labels):
    species=[]
    specPreds=[]    
    for index, pred in enumerate(preds.flatten()):
      st.write(f'{labels[index]}: {pred:.2f}%')
      species.append(labels[index])
      specPreds.append(pred)
    pred_df=pd.DataFrame()
    pred_df['Species']=species
    pred_df['Pred']=specPreds

    fig = px.bar(pred_df, x="Pred", y="Species", orientation='h',height=800, width=1000, title='NDBC Bees Identification')
    st.plotly_chart(fig)


    




  
  #Img preprocessing
  def preprocess_img(beeImgFile, IMG_SIZE=416):
    rawImg=image.load_img(beeImgFile, target_size=(IMG_SIZE,IMG_SIZE))
    imgArr=image.img_to_array(rawImg)
    imgArrExp = np.expand_dims(imgArr, 0)
    beeImg=tf.keras.applications.mobilenet_v3.preprocess_input(imgArrExp)
    return beeImg

if __name__ == '__main__':
    main()