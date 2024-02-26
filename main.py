import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
from keras.models import load_model
from utils import classify, set_background

set_background('./bg.jpg')

st.title('Pneumonia Classification')

st.header('Please uplaod a chest X-ray image')

file = st.file_uploader('Upload a file', type=['jpeg', 'jpg', 'png'])

model = load_model('./model/keras_model.h5')

with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    print(class_names)
    f.close()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    class_name, confidence_score = classify(image, model, class_names)

    st.write('## {}'.format(class_name))
    st.write('### score: {}'.format(confidence_score))
