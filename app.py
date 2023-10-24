# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
from ultralytics import YOLO
import time
import cv2
import settings



# Main page heading
st.title("Object Detection and Image Segementation using YOLOv8 in Python")

# Sidebar
st.sidebar.header("Machine Learning Model Configuration")

# Model Options
model_type = st.sidebar.radio(
    "Select the Task", ['Object Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence Score", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Object Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
col1, col2 = st.columns(2)
with col1:
    if source_img is not None:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
with col2:
    if st.sidebar.button('Detect Objects'):
        res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
        

