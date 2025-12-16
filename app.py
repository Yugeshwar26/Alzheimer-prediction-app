import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# 1. Configure the page
st.set_page_config(page_title="Alzheimer's Detector")

st.title("Alzheimer's Disease Detection (Debug Mode)")
st.write("Upload an MRI image. This version prints debug info to help fix errors.")

# 2. Download and Load Model with Size Check
@st.cache_resource
def load_my_model():
    model_path = 'MAIN.h5'
    file_id = '1WooTARsLQohA4LlipdewUs97kYR98OdK'
    url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
    
    # Check if file exists
    if not os.path.exists(model_path):
        st.write("⬇️ Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False, fuzzy=True)
    
    # Check file size (should be ~135MB)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    st.write(f"✅ Model file found. Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 100:
        st.error("⚠️ The model file is too small! It likely didn't download correctly. Please delete 'MAIN.h5' from your repo or try again.")
        return None

    return load_model(model_path)

# Load the model
try:
    with st.spinner('Loading Model...'):
        model = load_my_model()
        if model:
            st.success("Model loaded successfully!")
        else:
            st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Class Names
class_names = ['MILD', 'MODERATE', 'NON-DEMENTED', 'VERY MILD']

# 4. Image Upload
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded MRI', use_column_width=True)
        
        # 5. Preprocessing
        st.write("⚙️ Preprocessing image...")
        # Resize to 224x224 (match your notebook settings)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        st.write(f"Image shape: {img_array.shape}")

        # 6. Prediction
        if st.button("Analyze Image"):
            st.write("🧠 Running prediction...")
            prediction = model.predict(img_array)
            
            # Debug: Show raw probabilities
            st.write("Raw Prediction Probabilities:", prediction)
            
            index = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            result = class_names[index]
            
            st.success(f"**Result: {result}**")
            st.info(f"**Confidence: {confidence*100:.2f}%**")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        
