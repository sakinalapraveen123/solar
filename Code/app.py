import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "trained_effnet_finetune.h5"

st.set_page_config(
    page_title="Solar Panel Defect Scanner",
    page_icon="☀️",
    layout="wide"
)

# ---------- Top hero section ---------- #

st.markdown(
    """
    <style>
    .hero-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 0.98rem;
        color: #90939a;
        margin-bottom: 1.4rem;
    }
    .metric-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-right: 0.4rem;
        font-size: 0.8rem;
    }
    </style>
    <div class="hero-title">☀️ Solar Panel Defect Scanner</div>
    <div class="hero-subtitle">
        Upload a panel image and get a fast, AI‑powered health check powered by EfficientNet.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ---------- #

with st.sidebar:
    st.markdown("### 🧠 About this app")
    st.write(
        "This tool uses a **fine‑tuned EfficientNetB0** model to classify common solar panel "
        "conditions and defects in real time."
    )

    st.markdown("#### 🔎 It can detect:")
    st.write(
        "- Bird droppings\n"
        "- Clean panels\n"
        "- Dust accumulation\n"
        "- Electrical damage\n"
        "- Physical damage\n"
        "- Snow coverage"
    )

    st.markdown("---")
    st.markdown("#### 💡 Best results")
    st.write(
        "- Capture the full panel.\n"
        "- Avoid heavy glare and reflections.\n"
        "- Use clear, daytime images."
    )

    st.markdown("---")
    st.caption("Built by **Sakinala Praveen** · EfficientNet + Streamlit")

# ---------- Model loading ---------- #

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Pulling the latest AI model (only once)..."):
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

with st.spinner("Warming up the EfficientNet engine..."):
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

# ---------- Main layout ---------- #

left, right = st.columns([1.15, 1])

with left:
    st.markdown("### 📷 Step 1 · Upload your panel image")

    uploaded_file = st.file_uploader(
        "Drop a .jpg or .png here, or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Preview · Your uploaded image",
            use_column_width=True,
        )
    else:
        st.info("No image yet — upload a solar panel photo to get started.")

with right:
    st.markdown("### ⚙️ Step 2 · AI analysis")

    if uploaded_file is None:
        st.warning("Waiting for an image. Once you upload, the AI will generate a full report here.")
    else:
        # Preprocess
        img = image.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Scanning the panel for defects..."):
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        predicted_class = CLASSES[predicted_idx]

        # Main result card
        is_clean = predicted_class == "Clean"
        card_color = "#163c2e" if is_clean else "#582232"
        icon = "✅" if is_clean else "⚠️"

        st.markdown(
            f"""
            <div style="
                border-radius: 14px;
                padding: 1.1rem 1.25rem;
                margin-bottom: 0.8rem;
                background: linear-gradient(135deg, {card_color}, #000000);
                color: white;
                box-shadow: 0 6px 18px rgba(0,0,0,0.45);
            ">
                <div style="font-size: 1.7rem; font-weight: 600; margin-bottom: 0.2rem;">
                    {icon} {predicted_class}
                </div>
                <div style="font-size: 0.95rem; color: #e0e4ea; margin-bottom: 0.4rem;">
                    Model confidence: <b>{confidence:.1%}</b>
                </div>
                <div style="font-size: 0.86rem; color: #c5cad3;">
                    {"Panel health looks good — no obvious issues detected." if is_clean else
                    "The model suspects a defect or contamination. Consider cleaning or a detailed inspection."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top‑3 predictions
        st.markdown("#### 📊 Top confidence scores")

        top_indices = np.argsort(predictions)[-3:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            cls = CLASSES[idx]
            prob = float(predictions[idx])
            st.progress(min(prob, 1.0), text=f"{rank}. {cls} — {prob:.1%}")

# ---------- Extra details ---------- #

st.markdown("---")
st.markdown("### 🔬 Detailed class probabilities")

if "predictions" in locals():
    for cls, prob in zip(CLASSES, predictions):
        st.write(f"- **{cls}**: {prob:.1%}")
else:
    st.caption("Upload an image above to see a full probability breakdown.")
