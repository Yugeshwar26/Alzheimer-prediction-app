import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Configure the page
st.set_page_config(page_title="Alzheimer's Detector")

# 2. Title and Description
st.title("Alzheimer's Disease Detection")
st.write("Upload an MRI image to detect the stage of Alzheimer's Disease.")

# 3. Load the Model
@st.cache_resource
def load_my_model():
    return load_model('MAIN.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error("Error loading model. Make sure 'MAIN.h5' is uploaded to the repository.")
    st.stop()

# 4. Define Class Names
class_names = ['MILD', 'MODERATE', 'NON-DEMENTED', 'VERY MILD']

# 5. Image Upload
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI', use_column_width=True)
    
    # 6. Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 7. Make Prediction
    if st.button("Analyze Image"):
        prediction = model.predict(img_array)
        index = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        result = class_names[index]
        
        # 8. Show Result
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: {confidence*100:.2f}%")
      
