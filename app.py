import streamlit as st
import numpy as np
import tensorflow 
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
@st.cache_resource
def load_best_model():
    path = "C:/Users/artik/OneDrive/Desktop/fish_classification/env/Scripts/fish_claification_cnn.keras"
    model = load_model(path)
    return model

model = load_best_model()

fish_classes = ['animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize correctly
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand batch dimension
    return image

st.title("Fish Classifier App" "🦈")

st.text("This is a simple image classification web app to predict the fish species in the image.")
st.write("Upload an image of a fish and click on the 'Predict' button to see the prediction results.")
im = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.button("Predict")
if im is not None:
    image = Image.open(im)
    st.image(image, caption="Uploaded Image")
    im = preprocess_image(image)
    st.write("Prediction results")
    pred = fish_classes[np.argmax(model.predict(im))]
    
    st.write("Prediction: ", pred)
    st.write("Confidence: ", np.max(model.predict(im)) * 100, "%")
else:
    st.write("Please upload an image to classify.")

st.write("The app uses a pre-trained InceptionV3 model to predict the fish species.")
