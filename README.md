# Pneumonia Classification Project

![''](https://github.com/sirsjosh/Pneumonia-classification/blob/main/person5_bacteria_16.jpeg)

## Overview
The goal of this project is to leverage deep learning Neural Networks on Chest X-Ray images to determine which samples are from patients with Pneumonia.

## Dataset
The dataset used in this project consists of validated OCT (Optical Coherence Tomography) and Chest X-Ray images. The dataset is described and analyzed in the paper titled "Deep learning-based classification and referral of treatable human diseases." The OCT Images are divided into a training set and a testing set of independent patients. The OCT Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into four directories: CNV, DME, DRUSEN, and NORMAL.

### Dataset Link
[Download the dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

## Project Structure
- **model**: Contains the pre-trained Keras model for pneumonia classification (`keras_model.h5`).
- **utils**: Utility functions used in the project, including the image classification function (`classify`) and background setting function (`set_background`).
- **bg.jpg**: Background image for the Streamlit web application.
- **app.py**: Main Streamlit application file.
- **requirements.txt**: List of Python dependencies required for the project.

## How to Run
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit application using `streamlit run app.py`.
3. Open the provided URL in your web browser to interact with the pneumonia classification web app.

Feel free to explore the code and modify it according to your needs.
