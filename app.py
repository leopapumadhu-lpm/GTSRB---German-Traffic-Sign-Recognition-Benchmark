import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image to get its meaning and driving rule.")

# ----------------------------
# Load TFLite Model
# ----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="gtsrb_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 30

# ----------------------------
# GTSRB Sign Info
# ----------------------------
GTSRB_INFO = {
    0: 'Speed limit 20 km/h', 1: 'Speed limit 30 km/h', 2: 'Speed limit 50 km/h',
    3: 'Speed limit 60 km/h', 4: 'Speed limit 70 km/h', 5: 'Speed limit 80 km/h',
    6: 'End of speed limit 80 km/h', 7: 'Speed limit 100 km/h',
    8: 'Speed limit 120 km/h', 9: 'No passing',
    10: 'No passing > 3.5t', 11: 'Right-of-way',
    12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'No vehicles > 3.5t',
    17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows right',
    25: 'Road work', 26: 'Traffic signals',
    27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals', 32: 'End restrictions',
    33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End no passing > 3.5t'
}

# ----------------------------
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    name, desc, rule = GTSRB_INFO.get(
        class_id,
        ("Unknown sign", "No description available", "Proceed carefully")
    )

    st.success(f"🚦 **Sign:** {name}")
    st.info(f"📘 **Description:** {desc}")
    st.warning(f"🚗 **Driving Rule:** {rule}")
    st.metric("Confidence", f"{confidence:.2f}%")

else:
    st.warning("Please upload a traffic sign image.")
    st.title("🚦 Traffic Sign Recognition System")
    st.caption("Made by Madhuvanthi")
    

