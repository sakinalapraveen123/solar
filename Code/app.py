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
        Upload a rooftop shot and let my EfficientNet model give you a clean, no‑nonsense health report in a few seconds.
    </div>
    <div class="tag-row">
        <span class="tag-pill">EfficientNetB0 · Fine‑tuned</span>
        <span class="tag-pill">6 Defect Classes</span>
        <span class="tag-pill">Built by Sakinala Praveen</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 What this tool is")
    st.write(
        "SolarShield is a small project where I turned a **fine‑tuned EfficientNetB0** model "
        "into a simple panel health scanner. The goal is: upload once, get a quick, readable verdict."
    )

    st.markdown("#### 🪫 It currently detects")
    st.write(
        "- Bird droppings\n"
        "- Clean panels\n"
        "- Dust accumulation\n"
        "- Electrical damage\n"
        "- Physical damage\n"
        "- Snow coverage"
    )

    st.markdown("#### 📸 Before you upload")
    st.write(
        "- Try to capture the full panel, not just a corner.\n"
        "- Avoid extreme glare or very dark images.\n"
        "- Normal phone camera quality is perfectly fine."
    )

    st.markdown("---")
    st.caption(
        "Made as a B.Tech project by **Sakinala Praveen** · JNTUH\n\n"
        "SolarShield · AI that actually looks at your panels, not just your data."
    )

# -------------------------------------------------
# Load model (cached)
# -------------------------------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Pulling the latest model weights…"):
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

with st.spinner("Warming up the EfficientNet engine…"):
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
    st.markdown("### 📥 Step 1 · Drop a panel photo")

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
            "Tip: If the prediction feels off, try a clearer angle or better lighting."
        )
    else:
        st.info(
            "No image yet. Upload a clear rooftop solar panel photo to generate a health report."
        )

with right_col:
    st.markdown("### ⚡ Step 2 · Let SolarShield inspect it")

    if uploaded_file is None:
        st.warning("Waiting for an image. Once you upload, the AI verdict will appear here.")
        predictions = None
    else:
        # preprocess
        img = image.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Scanning surface patterns, textures and hotspots…"):
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        predicted_class = CLASSES[predicted_idx]

        is_clean = predicted_class == "Clean"
        card_color = "#14532d" if is_clean else "#7f1d1d"
        icon = "✅" if is_clean else "⚠️"

        summary_text = (
            "Panel looks healthy. No major issues flagged by the model."
            if is_clean
            else "Something looks off. Consider a closer inspection, cleaning, or a technician visit."
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
# Full breakdown + model story
# -------------------------------------------------
st.markdown("---")
cols_bottom = st.columns([1.2, 1])

with cols_bottom[0]:
    st.markdown("### 🔬 Class probabilities")

    if "predictions" in locals() and predictions is not None:
        for cls, prob in zip(CLASSES, predictions):
            st.write(f"- **{cls}** · {prob:.1%}")
    else:
        st.caption("Upload an image above to see the full probability breakdown for every class.")

with cols_bottom[1]:
    st.markdown("### 🧬 Under the hood")
    st.write(
        "This model started as **EfficientNetB0** pre‑trained on ImageNet. "
        "I fine‑tuned it on a curated dataset of solar panel images covering dust, snow, "
        "bird droppings, and multiple damage types. I used Keras Tuner to play with "
        "learning rate, dropout, and dense layer size, and deployed the best checkpoint "
        "here as a small, real‑time demo."
    )
