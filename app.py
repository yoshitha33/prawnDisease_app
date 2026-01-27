import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Prawn Disease Classification", layout="centered")

st.title("Prawn Disease Classification")
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
# Load MediaPipe for human detection
# ---------------------------
@st.cache_resource
def load_cascade_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade_classifier()

# ---------------------------
# Human Detection Function
# ---------------------------
def detect_human(image):
    """
    Detect if the image contains a human using face detection
    Returns: (has_human, human_confidence)
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        return True, 0.9
    return False, 0.0

# ---------------------------
# Prawn Detection Function
# ---------------------------
def detect_prawn(image):
    """
    Detect if the image contains a prawn using color and morphology analysis
    Returns: (is_prawn, confidence_score)
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for prawn (orange, pink, brown, gray tones)
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    lower_pink = np.array([140, 30, 100])
    upper_pink = np.array([180, 200, 255])
    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 50, 200])
    
    # Create masks
    mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
    mask_pink = cv2.inRange(img_hsv, lower_pink, upper_pink)
    mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(mask_orange, mask_pink)
    combined_mask = cv2.bitwise_or(combined_mask, mask_gray)
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate prawn-like color coverage
    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
    prawn_pixels = np.sum(combined_mask > 0)
    coverage = prawn_pixels / total_pixels
    
    # Prawn typically occupies 10-85% of image
    is_prawn = 0.10 < coverage < 0.85
    confidence = min(coverage * 1.5, 1.0) if is_prawn else 0.0
    
    return is_prawn, confidence

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
    # Check if human is detected
    # ---------------------------
    has_human, human_confidence = detect_human(image)
    
    if has_human:
        st.error("❌ NOT A PRAWN IMAGE!")
        st.write("Please upload only prawn images without humans for disease classification.")
    else:
        # ---------------------------
        # Check if prawn is detected
        # ---------------------------
        is_prawn, prawn_confidence = detect_prawn(image)
        
        if not is_prawn:
            st.error("❌ NOT A PRAWN IMAGE!")
            st.write("This is not a prawn. Please upload a prawn image only.")
        else:
            st.success("✅ Prawn detected! Analyzing for disease...")
            
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
print("App is running...")