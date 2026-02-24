import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Skin Cancer AI Detection", page_icon="🩺", layout="wide")

# 2. Sidebar Configuration (For Business Context)
with st.sidebar:
    st.title("🩺 Project Context")
    st.info("This Deep Learning for Managers (DLM) project demonstrates how AI can assist in the early detection of lifestyle-aggravated diseases.")
    
    st.markdown("### 📊 Strategic Value")
    st.markdown("- **Automated Triage:** Accelerates initial patient screening.\n- **Cost Efficiency:** Reduces operational bottlenecks.\n- **Decision Support:** Provides data-backed secondary insights for medical staff.")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This is an academic prototype for analytical demonstration. It is not intended to replace professional medical diagnosis.")

# 3. Main Dashboard Header
st.title("Lifestyle Disease Analytics: Skin Cancer Detection")
st.markdown("Leveraging **Convolutional Neural Networks (CNN)** to classify skin lesions through automated visual pattern recognition.")
st.markdown("---")

# 4. Load the AI Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_cancer_cnn.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load the model. Error details: {e}")
    st.stop()

# 5. Dashboard Layout (Two Columns)
col1, col2 = st.columns([1, 1]) # Splits the screen 50/50

# LEFT COLUMN: Image Upload
with col1:
    st.subheader("1. Input Data")
    st.write("Upload a clear, standardized image of the skin lesion.")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Target Image for Analysis', use_column_width=True)

# RIGHT COLUMN: AI Analysis & Metrics
with col2:
    st.subheader("2. AI Diagnostics")
    
    if uploaded_file is None:
        st.info("👈 Upload an image on the left to generate AI analysis and confidence metrics.")
    else:
        # A prominent, colored button
        if st.button("🔍 Run Neural Network Analysis", type="primary"):
            with st.spinner("Extracting image features and running analysis..."):
                
                # Preprocess the image
                img = image.resize((150, 150))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make Prediction
                prediction = model.predict(img_array)
                score = prediction[0][0] # The raw probability score from the sigmoid layer
                
                st.markdown("### Analytical Results")
                
                # Logic to display Confidence Metrics
                if score > 0.5:
                    confidence = score * 100
                    st.error("⚠️ **High Risk:** The model detected characteristics visually indicative of a **Malignant** lesion.")
                    # Display metric and progress bar
                    st.metric(label="AI Confidence Score (Malignant)", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))
                else:
                    confidence = (1 - score) * 100
                    st.success("✅ **Low Risk:** The model detected characteristics visually indicative of a **Benign** lesion.")
                    # Display metric and progress bar
                    st.metric(label="AI Confidence Score (Benign)", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))
                
                st.markdown("---")
                st.write("**Recommended Action:** Regardless of the AI output, any suspicious or evolving skin lesions should be evaluated by a certified dermatologist.")
