import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/MobileNetV3Large.keras')

# Define image size (same as what your model expects)
image_size = (224, 224)

# Define class names (in the same order as during training)
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
This is a **Garbage Classifier** using a MobileNetV3Large model trained on 6 types of garbage categories.
- Built with 🧠 TensorFlow
- UI by 🚀 Streamlit
""")

st.sidebar.markdown("**Developed by:** Aman Kumar")
st.sidebar.markdown("---")
st.sidebar.info("Upload an image of trash to predict the type.")

# Main app
st.title("Garbage Classifier")
st.write("Upload an image and I will predict the type of garbage.")

# Camera input
camera_photo = st.camera_input("Take a photo")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image = None
if camera_photo is not None:
    image = Image.open(camera_photo).convert('RGB')
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
if image:

    # Preprocess the image
    img = image.resize(image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    # Predict with spinner
    with st.spinner("🔍 Analyzing image..."):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = 100 * np.max(predictions[0])

    # Layout with 2 columns
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded Image", use_container_width=True)
    col2.markdown(f"### Prediction: **{predicted_class}**")
    col2.markdown(f"**Confidence:** {confidence:.2f}%")

    # Progress bar
    st.markdown(f"**Prediction Confidence:**")
    st.progress(float(confidence / 100))
