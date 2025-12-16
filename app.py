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

# 2. Title and Description
st.title("Alzheimer's Disease Detection")
st.write("Upload an MRI image to detect the stage of Alzheimer's Disease.")

# 3. Download and Load the Model
@st.cache_resource
def load_my_model():
    model_path = 'MAIN.h5'
    
    # Check if model exists, if not download it from Google Drive
    if not os.path.exists(model_path):
        # Your specific File ID
        file_id = '1WooTARsLQohA4LlipdewUs97kYR98OdK'
        
        # Use the full URL format which works reliably with fuzzy=True
        url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
        
        # fuzzy=True is REQUIRED to bypass the "file too large for virus scan" warning
        gdown.download(url, model_path, quiet=False, fuzzy=True)
    
    return load_model(model_path)

try:
    with st.spinner('Loading Model... (This may take a minute)'):
        model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. Define Class Names
# Ensure these match the order from your training notebook
class_names = ['MILD', 'MODERATE', 'NON-DEMENTED', 'VERY MILD']

# 5. Image Upload
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI', use_column_width=True)
    
    # 6. Preprocess the image
    # Resize to the same size used in training (224x224)
    img = img.resize((224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dims to match batch shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale pixel values (1./255)
    img_array /= 255.0

    # 7. Make Prediction
    if st.button("Analyze Image"):
        prediction = model.predict(img_array)
        # Get the index of the highest probability
        index = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        result = class_names[index]
        
        # 8. Show Result
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: {confidence*100:.2f}%")
        
