import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Set up the page appearance
st.set_page_config(page_title="Skin Cancer Detection", page_icon="⚕️")
st.title("Lifestyle Disease Analytics: Skin Cancer Detection")
st.write("Upload a skin lesion image to evaluate if it is visually indicative of Benign or Malignant characteristics.")

# 2. Load the trained Keras model safely
@st.cache_resource
def load_model():
    # Looking exclusively for the new .keras file we made
    return tf.keras.models.load_model('skin_cancer_cnn.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load the model. Please ensure 'skin_cancer_cnn.keras' is uploaded to GitHub. Error details: {e}")
    st.stop()

# 3. Create the file uploader
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 4. Preprocess the image to exactly match the Colab training data
    # Resize to 150x150 pixels
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add the batch dimension
    
    # 5. Create an analyze button and make the prediction
    if st.button('Analyze Image'):
        with st.spinner('Running AI Analysis...'):
            prediction = model.predict(img_array)
            
            # Since we used a sigmoid activation, a score > 0.5 is Class 1 (Malignant)
            if prediction[0][0] > 0.5:
                st.error("⚠️ Alert: Model detects potential Malignant characteristics. Professional medical review recommended.")
            else:
                st.success("✅ Result: Model detects Benign characteristics.")
