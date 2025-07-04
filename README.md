# Garbage Classifier App ♻️

A deep learning-based web application to classify images of garbage into categories like **Cardboard, Glass, Metal, Paper, Plastic**, and **Trash** using transfer learning with MobileNetV3Large.

##  Demo
Upload an image of waste and get an instant prediction of its category, along with confidence level.

##  Features
- Built with **TensorFlow** and **Keras**
- Uses **MobileNetV3Large** for high accuracy with fast inference
- Clean, responsive **Streamlit** interface
- Displays prediction and confidence visually
- Option to upload images and get real-time results

##  Classes
The model is trained to identify the following classes:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

##  Model Architecture
- Base model: `MobileNetV3Large` (pretrained on ImageNet)
- Input size: `224x224x3`
- Fine-tuned top layers
- Accuracy achieved: **94.38%**
