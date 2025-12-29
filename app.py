import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# ---------------- LOAD MODEL AND CLASSES ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()


# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 App Controls")
st.sidebar.markdown("""
**Model Type:** CNN (.h5)
**Task:** Image Classification
**Input Size:** 30 × 30
""")

st.sidebar.info("Upload a traffic sign image to predict its class.")

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align: center;'>🚦 Traffic Sign Recognition System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>"
    "AI-powered application using Deep Learning to classify traffic signs"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1, 1])

# ---------------- LEFT COLUMN ----------------
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------------- RIGHT COLUMN ----------------
with col2:
    st.subheader("📊 Prediction Result")

    if uploaded_file:
        # Preprocessing
        # Resize to 32x32 based on model training, not 30x30 as commented
        img = image.resize((32, 32)) 
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Get class name
        class_name = classes.get(class_id, "Unknown")

        st.markdown(f"<div style='background-color: #e6f7ff; padding: 15px; border-radius: 10px;'>"  # Light blue background
                    f"<h4 style='color: #0056b3;'>Predicted Traffic Sign:</h4>"
                    f"<h3 style='color: #007bff;'>**{class_name}**</h3>"
                    f"<p style='font-size: 16px;'>Confidence: **{confidence:.2f}%**</p>"
                    f"</div>", unsafe_allow_html=True)
