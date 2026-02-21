import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ Solar Panel Defect Classifier")
st.write("Upload an image of a solar panel to detect defects using your trained EfficientNet model.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("solar_panel_classifier.h5")
    return model

with st.spinner("Loading model..."):
    model = load_model()
