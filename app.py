import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Prawn Disease Classification", layout="centered")

st.title("🦐 Prawn Disease Classification")
st.write("Upload a prawn image to detect the disease (BG, BG-WSSV, or WSSV).")

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("prawn_disease_model_final.keras")
    return model

model = load_model()

# ---------------------------
# Hard-set class order
# ---------------------------
class_names = ['BG', 'BG-WSSV', 'WSSV']

# ---------------------------
# Image uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a prawn image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------------------
    # Preprocess (MUST match training)
    # ---------------------------
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # ---------------------------
    # Predict
    # ---------------------------
    preds = model.predict(img_array)[0]

    st.subheader("🔍 Class Probabilities")
    for name, score in zip(class_names, preds):
        st.write(f"{name}: {score:.4f}")

    idx = np.argmax(preds)
    confidence = preds[idx]

    st.success(f"✅ Predicted Disease: {class_names[idx]}")



    
    st.write(f"📊 Confidence: {confidence * 100:.2f}%")
