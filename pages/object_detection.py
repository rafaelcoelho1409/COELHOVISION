import streamlit as st
from streamlit_extras.grid import grid
import numpy as np
import cv2
import tensorflow as tf
from ultralytics.utils.plotting import Annotator
from functions import (
    option_menu,
    page_buttons,
    image_border_radius,
    Hex_to_RGB
)
from functions import (
    FullFaceDetector,
    ObjectDetectionYOLO,
    MediaPipeObjectDetection,
    MediaPipeImageClassification,
    MediaPipeFaceDetector
)

st.set_page_config(
    page_title = "COELHO VISION | Object Detection",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

grid_title = grid([5, 1], vertical_align = True)
container1 = grid_title.container()
container1.title("$$\\large{\\textbf{COELHO VISION}}$$")
container1.write("$$\\Huge{\\textit{\\textbf{Object Detection}}}$$")
container1.caption("Author: Rafael Silva Coelho")

page_buttons()

st.divider()
image_border_radius("./assets/coelho_vision_logo.png", 20, 100, 100, grid_title)

grid_filters = grid(2, vertical_align = True)
mode_filter = grid_filters.selectbox(
    label = "Mode",
    options = [
        "Camera",
        "Image"
    ]
)
role_filter = grid_filters.selectbox(
    label = "Role",
    options = [
        "Full Face Detection",
        "Image Classification (VGG16)",
        "Object Detection (YOLOv8)",
        "Object Detection (MediaPipe)",
        "Image Classification (MediaPipe)",
        "Face Detector (MediaPipe)"
    ]
)

grid_display = grid([1, 2], vertical_align = True)
grid1 = grid_display.container()
grid2 = grid_display.container()
if mode_filter == "Image":
    uploaded_image = grid1.file_uploader(
        label = "Upload image",
        type = ["jpg", "jpeg", "png", "webp"],
        accept_multiple_files = False
    )
    if uploaded_image is not None:
        image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
elif mode_filter == "Camera":
    cam_image = grid1.camera_input("Teste")
    if cam_image is not None:
        image_bytes = np.asarray(bytearray(cam_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)


if role_filter == "Full Face Detection":
    detector = FullFaceDetector()
    grid1_cols = grid1.columns(2)
    face_scaleFactor = grid1_cols[0].slider(
        label = "Face scale factor",
        min_value = 1.00,
        max_value = 3.00,
        value = 1.50,
        step = 0.01
    )
    eye_scaleFactor = grid1_cols[0].slider(
        label = "Eye scale factor",
        min_value = 1.00,
        max_value = 3.00,
        value = 1.30,
        step = 0.01
    )
    mouth_scaleFactor = grid1_cols[0].slider(
        label = "Mouth scale factor",
        min_value = 1.00,
        max_value = 3.00,
        value = 3.00,
        step = 0.01
    )
    face_minNeighbors = grid1_cols[1].slider(
        label = "Face min. neighbors",
        min_value = 1,
        max_value = 10,
        value = 5,
        step = 1
    )
    eye_minNeighbors = grid1_cols[1].slider(
        label = "Eye min. neighbors",
        min_value = 1,
        max_value = 10,
        value = 7,
        step = 1
    )
    mouth_minNeighbors = grid1_cols[1].slider(
        label = "Mouth min. neighbors",
        min_value = 1,
        max_value = 10,
        value = 10,
        step = 1
    )
    grid2.header(role_filter)
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        detector.face_scaleFactor = face_scaleFactor
        detector.eye_scaleFactor = eye_scaleFactor
        detector.mouth_scaleFactor = mouth_scaleFactor
        detector.face_minNeighbors = face_minNeighbors
        detector.eye_minNeighbors = eye_minNeighbors
        detector.mouth_minNeighbors = mouth_minNeighbors
        transformed_image = detector.transform(opencv_image)
        grid2.image(transformed_image, use_column_width = True)
elif role_filter == "Image Classification (VGG16)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = tf.keras.applications.vgg16.VGG16(
            weights = "imagenet",
            include_top = True
        )
        tensor_img = tf.convert_to_tensor(opencv_image)
        tensor_reshaped = tf.image.resize(tensor_img, (224, 224))
        model.compile(
            optimizer = "RMSprop",
            loss = "categorical_crossentropy"
        )
        tensor_pred = tf.expand_dims(tensor_reshaped, axis = 0)
        tensor_pred = tf.keras.applications.imagenet_utils.preprocess_input(tensor_pred)
        probabilities = model.predict(tensor_pred)
        P = tf.keras.applications.imagenet_utils.decode_predictions(probabilities)
        grid2.header(role_filter)
        grid2.subheader("Predictions")
        grid2_cols = grid2.columns(5)
        for i in range(5):
            grid2_cols[i].metric(
                label = P[0][i][1].replace("_", " ").title(),
                value = "{:.2f}%".format(P[0][i][2] * 100)
            )
        grid2.image(opencv_image, use_column_width = True)
elif role_filter == "Object Detection (YOLOv8)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = ObjectDetectionYOLO()
        results = model.model.predict(opencv_image)
        for r in results:    
            annotator = Annotator(opencv_image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.model.names[int(c)], color = (255, 0, 0), txt_color = (255, 255, 255))      
        img = annotator.result()
        grid2.header(role_filter)
        grid2.image(img, use_column_width = True)
elif role_filter == "Object Detection (MediaPipe)":
    text_color_picker = grid1.color_picker(
        label = "Set a color to bounding boxes and text"
    )
    text_color = Hex_to_RGB(text_color_picker[1:])
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipeObjectDetection()
        model.text_color = text_color
        img = model.transform(opencv_image)
        grid.subheader(role_filter)
        grid2.image(img, use_column_width = True)
elif role_filter == "Image Classification (MediaPipe)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipeImageClassification()
        grid2.subheader(role_filter)
        results = model.transform(opencv_image)
        grid2_cols = grid2.columns(len(results.classifications[0].categories))
        for i in range(len(results.classifications[0].categories)):
            grid2_cols[i].metric(
                label = results.classifications[0].categories[i].category_name.title(),
                value = "{:.2f}%".format(results.classifications[0].categories[i].score * 100)
            )
        grid2.image(opencv_image, use_column_width = True)
elif role_filter == "Face Detector (MediaPipe)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipeFaceDetector()
        grid2.subheader(role_filter)
        img = model.transform(opencv_image)
        grid2.image(img, use_column_width = True)