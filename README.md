# SolarShield · Solar Panel Defect Classification ☀️

**SolarShield** is an end‑to‑end deep learning application for automatic solar panel defect classification from images.  
It combines a fine‑tuned EfficientNetB0 model with a modern Streamlit UI to provide fast, interpretable panel health assessments.

🔗 **Live Demo**: https://solar-panel-detect.streamlit.app/  

---

## 1. Overview

Solar photovoltaic (PV) systems are sensitive to surface conditions such as dust, bird droppings, physical cracks, and snow.  
Routine manual inspection is time‑consuming, subjective, and difficult to scale for large installations.  

**Goal of this project:**  
Build an image‑based system that can quickly classify the condition of a solar panel into predefined defect categories and present the result through a clean web interface suitable for academic and practical demonstrations.

---

## 2. Features

- 📸 **Image upload interface**  
  Upload a rooftop solar panel image directly from the browser.

- 🧠 **Deep learning‑based classification**  
  Fine‑tuned **EfficientNetB0** backbone for robust feature extraction.

- 🔎 **Six condition classes**
  - Clean  
  - Dusty  
  - Bird‑drop  
  - Electrical‑damage  
  - Physical‑Damage  
  - Snow‑Covered  

- 📊 **Rich feedback**
  - Predicted class with confidence score  
  - Top‑3 class probabilities with progress bars  
  - Full probability distribution across all classes  

- 🖥️ **Streamlit UI**
  - Dark, distraction‑free layout  
  - Two‑step workflow: *Upload → Inspect*  
  - Suitable for classroom presentations, demos, and viva.

---

## 3. System Architecture

**High‑level pipeline:**

1. User uploads a solar panel image through the Streamlit app.  
2. The image is:
   - Converted to RGB  
   - Resized to \(224 \times 224\)  
   - Normalized using EfficientNet’s `preprocess_input`.  
3. The preprocessed tensor is passed to a fine‑tuned **EfficientNetB0** model.  
4. The model outputs class probabilities (softmax over 6 classes).  
5. The UI displays:
   - Final predicted class and confidence  
   - Top‑3 class scores  
   - Detailed breakdown for all classes.

This design cleanly separates **presentation (Streamlit)** and **inference (TensorFlow/Keras)** while remaining simple enough for an academic project.

---

## 4. Model Details

- **Base architecture:** EfficientNetB0 (pre‑trained on ImageNet)  
- **Task type:** Multi‑class image classification (single label per image)  
- **Input size:** 224 × 224 × 3  
- **Loss function:** Categorical Cross‑Entropy  
- **Primary metric:** Accuracy  

**Fine‑tuning strategy:**

- Freeze early EfficientNet layers initially, train custom classification head.  
- Gradually unfreeze selected deeper blocks for fine‑tuning.  
- Use **Keras Tuner** (or manual search) to explore:
  - Learning rate  
  - Dropout rate  
  - Size of dense layers  

This approach provides a good trade‑off between performance and training time on academic‑scale hardware.

---

## 5. Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras (EfficientNetB0)  
- **Web Framework:** Streamlit  
- **Image Handling:** Pillow (PIL), NumPy  
- **Model Persistence:** `.h5` model file (optionally fetched from Google Drive)  

---

## 6. Project Structure

Example folder layout (adapt to your repo):

```bash
.
├── Code/
│   └── app.py                       # Streamlit application
├── models/
│   └── trained_effnet_finetune.h5   # Trained EfficientNet model (optional, can be downloaded)
├── data/                            # (optional) Sample images for testing
├── .streamlit/
│   └── config.toml                  # Theme configuration (dark UI)
├── requirements.txt
└── README.md
7. Getting Started
7.1. Clone the repository
bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
7.2. (Optional) Create a virtual environment
bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
7.3. Install dependencies
bash
pip install -r requirements.txt
7.4. Run the Streamlit app
bash
cd Code
streamlit run app.py
Open the URL shown in the terminal (typically http://localhost:8501) in your browser.

If the model file is not found locally, app.py can be configured to download it automatically from Google Drive before inference.

8. Usage Workflow
Open the app (locally or via the Streamlit Cloud link).

Upload an image of a solar panel (JPG/PNG).

Wait for the model to analyze the image.

View:

The predicted condition label.

Confidence score.

Top‑3 classes and full probability table.

Optionally repeat with different images (dusty, damaged, etc.) to observe behavior.

9. Possible Improvements
This project is intentionally kept focused and interpretable.
Future enhancements may include:

Training on a larger, more diverse dataset (different regions, seasons, capture devices).

Exploring newer architectures (EfficientNetV2, ConvNeXt, Vision Transformers).

Combining image‑based predictions with sensor data (voltage, current, temperature).

Generating maintenance reports and trend analysis over time.

Deploying behind an authentication layer for use in real installations.

10. Academic Context
SolarShield is designed to fit well as a B.Tech / B.E. mini‑project or real‑time research project:

Demonstrates application of transfer learning to a real‑world problem.

Shows a complete pipeline from data → model → deployment → UI.

Uses tools and frameworks (TensorFlow, Streamlit) that are widely accepted in academia and industry.

If you use this for a presentation or viva, you can highlight:

Why automated visual inspection of solar panels is useful.

How EfficientNet is leveraged for efficient, accurate classification.

How Streamlit simplifies deployment and interaction with the model.

11. Live Demo
👉 Try it here: https://solar-panel-detect.streamlit.app/

Upload any clear solar panel image and see the predicted condition in real time.
