import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------------------------
# Basic config
# -------------------------------------------------
MODEL_PATH = "trained_effnet_finetune.h5"

st.set_page_config(
    page_title="SolarShield · Panel Health Scanner",
    page_icon="☀️",
    layout="wide",
)

# -------------------------------------------------
# Global styling (CSS)
# -------------------------------------------------
st.markdown(
    """
    <style>
    header, footer {visibility: hidden;}

    .main > div {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }

    .hero-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.2rem;
    }

    .tag-row {
        text-align: center;
        margin-bottom: 1.6rem;
    }

    .tag-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        margin: 0 0.25rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: #e5e7eb;
        font-size: 0.75rem;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(8px);
    }

    .note-text {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    </style>

    <div class="hero-title">
        ☀️ SolarShield · Panel Health Scanner
    </div>
    <div class="hero-subtitle">
        Upload a rooftop shot and let an EfficientNet model give you a clear panel health report in a few seconds.
    </div>
    <div class="tag-row">
        <span class="tag-pill">EfficientNetB0 · Fine‑tuned</span>
        <span class="tag-pill">6 Defect Classes</span>
        <span class="tag-pill">Real‑time Inference</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Application overview")
    st.write(
        "This application demonstrates how a **fine‑tuned EfficientNetB0** model can be used "
        "to classify the condition of a solar panel from a single image. It is designed as an "
        "academic project to showcase end‑to‑end model deployment."
    )

    st.markdown("#### 🪫 Classes covered")
    st.write(
        "- Bird droppings\n"
        "- Clean panels\n"
        "- Dust accumulation\n"
        "- Electrical damage\n"
        "- Physical damage\n"
        "- Snow coverage"
    )

    st.markdown("#### 📸 Image guidelines")
    st.write(
        "- Capture as much of the panel surface as possible.\n"
        "- Avoid extreme glare or very low‑light images.\n"
        "- Standard mobile camera quality is sufficient."
    )

    st.markdown("---")
    st.caption(
        "SolarShield · A Streamlit demonstration of solar panel defect classification using EfficientNet."
    )

# -------------------------------------------------
# Load model (cached)
# -------------------------------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights…"):
            import subprocess
            subprocess.run(["pip", "install", "-q", "gdown"], check=True)
            import gdown

            gdrive_url = (
                "https://drive.google.com/uc?"
                "id=1KNPbVWWTArb3d0eegs2gIgsKViZn7hK0&export=download"
            )
            gdown.download(gdrive_url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

with st.spinner("Loading EfficientNet model into memory…"):
    model = download_and_load_model()

# -------------------------------------------------
# Constants
# -------------------------------------------------
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

# -------------------------------------------------
# Layout
# -------------------------------------------------
left_col, right_col = st.columns([1.08, 1])

with left_col:
    st.markdown("### 📥 Step 1 · Upload a panel image")

    uploaded_file = st.file_uploader(
        "Drag & drop a .jpg or .png file, or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Preview · Uploaded solar panel",
            use_column_width=True,
        )
        st.caption(
            "If the prediction appears unreliable, try uploading a clearer image with better lighting."
        )
    else:
        st.info(
            "Upload a rooftop solar panel image to generate an automated health assessment."
        )

with right_col:
    st.markdown("### ⚡ Step 2 · Run the model")

    if uploaded_file is None:
        st.warning("Waiting for an image. Once you upload, the model output will be shown here.")
        predictions = None
    else:
        # Preprocess
        img = image.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Analyzing panel surface patterns and textures…"):
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        predicted_class = CLASSES[predicted_idx]

        is_clean = predicted_class == "Clean"
        card_color = "#14532d" if is_clean else "#7f1d1d"
        icon = "✅" if is_clean else "⚠️"

        summary_text = (
            "The panel appears to be in a healthy condition based on the model output."
            if is_clean
            else "The model indicates possible contamination or damage. Further inspection is recommended."
        )

        st.markdown(
            f"""
            <div style="
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                margin-bottom: 1.1rem;
                background: radial-gradient(circle at top left, {card_color}, #020617);
                color: white;
                box-shadow: 0 12px 30px rgba(0,0,0,0.55);
            ">
                <div style="font-size: 1.7rem; font-weight: 700; margin-bottom: 0.2rem;">
                    {icon} {predicted_class}
                </div>
                <div style="font-size: 0.98rem; color: #e5e7eb; margin-bottom: 0.4rem;">
                    Model confidence: <b>{confidence:.1%}</b>
                </div>
                <div style="font-size: 0.9rem; color: #d1d5db;">
                    {summary_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Top 3 class scores")

        top_indices = np.argsort(predictions)[-3:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            cls = CLASSES[idx]
            prob = float(predictions[idx])
            st.progress(
                min(prob, 1.0),
                text=f"{rank}. {cls} — {prob:.1%}",
            )

# -------------------------------------------------
# Full breakdown + model description
# -------------------------------------------------
st.markdown("---")
cols_bottom = st.columns([1.2, 1])

with cols_bottom[0]:
    st.markdown("### 🔬 Class probabilities")

    if "predictions" in locals() and predictions is not None:
        for cls, prob in zip(CLASSES, predictions):
            st.write(f"- **{cls}** · {prob:.1%}")
    else:
        st.caption("Upload an image above to view the full probability distribution across all classes.")

with cols_bottom[1]:
    st.markdown("### 🧬 Model details")
    st.write(
        "The backbone of this application is **EfficientNetB0** pre‑trained on ImageNet and then "
        "fine‑tuned on a curated dataset of solar panel images covering dust, snow, bird droppings, "
        "and different damage types. Hyperparameters such as learning rate, dropout, and dense layer "
        "size were explored using Keras Tuner, and the best performing checkpoint was deployed here "
        "to enable real‑time inference through a Streamlit interface."
    )
