# Lifestyle Disease Analytics: AI-Driven Skin Cancer Detection ⚕️

## Project Overview
This repository contains the deployment files for a Deep Learning prototype aimed at early detection of lifestyle-aggravated diseases. Specifically, this application utilizes a Convolutional Neural Network (CNN) to analyze unstructured image data (retinal or skin scans) and classify lesions as either visually indicative of **Benign** or **Malignant** characteristics. 

## Strategic Business Value
Bridging the gap between raw data and actionable healthcare solutions, this project demonstrates how organizations can leverage predictive analytics to:
* **Automate Triage:** Reduce the initial screening burden on medical professionals by filtering high-risk cases.
* **Cost Optimization:** Lower the operational costs associated with manual, broad-spectrum diagnostic processes.
* **Enhance Decision-Making:** Provide data-backed, rapid secondary opinions to healthcare providers to improve patient outcomes.

## Technical Stack
* **Model Training:** Google Colab, TensorFlow, Keras
* **Architecture:** Custom Convolutional Neural Network (CNN) for image edge and pattern detection
* **Web Deployment:** Streamlit, Python
* **Data Processing:** NumPy, Pillow (PIL)

## Repository Structure
* `skin_cancer_cnn.keras`: The trained and compiled deep learning model.
* `app.py`: The Streamlit Python script that generates the interactive web interface.
* `requirements.txt`: The dependency list required for the cloud server to run the application.

## How to Use the Application
1. Access the live Streamlit web application via the deployment link.
2. Upload a clear, standardized image of a skin lesion (.jpg, .jpeg, or .png).
3. Click **Analyze Image**. 
4. The CNN will process the image array and output a predictive classification based on learned malignant vs. benign patterns.

*Disclaimer: This is an academic prototype for analytical demonstration purposes. It is not intended to replace professional medical diagnosis.*
