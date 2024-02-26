# Import necessary libraries
import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
from keras.models import load_model
from utils import classify, set_background

# Set the background image
set_background('./bg.jpg')

# Set the title and header for the Streamlit app
st.title('Pneumonia Classification')
st.header('Please upload a chest X-ray image')

# Allow the user to upload a file with specified types
file = st.file_uploader('Upload a file', type=['jpeg', 'jpg', 'png'])

# Load the pre-trained Keras model for pneumonia classification
model = load_model('./model/keras_model.h5')

# Read class names from the labels file
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Check if a file has been uploaded
if file is not None:
    # Open and display the uploaded image
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify the image using the loaded model
    class_name, confidence_score = classify(image, model, class_names)

    # Display the predicted class and confidence score
    st.write('## {}'.format(class_name))
    st.write('### Score: {}'.format(confidence_score))
