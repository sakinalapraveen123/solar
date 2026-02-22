import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "trained_effnet_finetune.h5"

st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ Solar Panel Defect Classifier")
st.write("Upload an image of a solar panel to detect defects using your tuned EfficientNet model.")

@st.cache_resource
def download_and_load_model():
    # Download model from Google Drive if not present
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model file (first run might take some time)...")

        import subprocess
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        import gdown

        # Your Google Drive file (trained_effnet_finetune.h5)
        gdrive_url = "https://drive.google.com/uc?id=1KNPbVWWTArb3d0eegs2gIgsKViZn7hK0&export=download"

        gdown.download(gdrive_url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

with st.spinner("Loading model..."):
    model = download_and_load_model()

CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]

IMG_HEIGHT = 224
IMG_WIDTH = 224

uploaded_file = st.file_uploader(
    "Upload a solar panel image..",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    with st.spinner("Analyzing the panel..."):
        predictions = model.predict(img_array, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    predicted_class = CLASSES[predicted_idx]

    st.markdown(
        f"### **Prediction: {predicted_class}**\n"
        f"***Confidence: {confidence:.1%}**"
    )

    if predicted_class == "Clean":
        st.success("The panel appears to be in good condition!!")
    else:
        st.warning("A defect or contamination has been detected!!")

    st.write("### Top 3 Predictions")
    top_indices = np.argsort(predictions)[-3:][::-1]

    for i, idx in enumerate(top_indices):
        class_name = CLASSES[idx]
        prob = float(predictions[idx])
        if i == 0:
            st.markdown(f"**1st**: **{class_name}** - {prob:.1%}")
        elif i == 1:
            st.markdown(f"**2nd**: {class_name} - {prob:.1%}")
        else:
            st.markdown(f"**3rd**: {class_name} - {prob:.1%}")

    with st.expander("View all class probabilities"):
        for i, prob in enumerate(predictions):
            st.write(f"{CLASSES[i]:<20} {prob:.1%}")
