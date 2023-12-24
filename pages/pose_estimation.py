import streamlit as st
import numpy as np
import cv2
from streamlit_extras.grid import grid
from functions import (
    option_menu,
    page_buttons,
    image_border_radius
)
from functions import (
    MediaPipeGestureRecognition,
    MediaPipeHandLandmarker,
    MediaPipePoseEstimation
)

st.set_page_config(
    page_title = "COELHO VISION | Pose Estimation",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

grid_title = grid([5, 1], vertical_align = True)
container1 = grid_title.container()
container1.title("$$\\large{\\textbf{COELHO VISION | Pose Estimation}}$$")
container1.caption("Author: Rafael Silva Coelho")

page_buttons()

st.divider()
image_border_radius("./assets/coelho_vision_logo.png", 20, 100, 100, grid_title)

grid_filters = grid(2, vertical_align = True)
mode_filter = grid_filters.selectbox(
    label = "Mode",
    options = [
        "Image",
        "Camera"
    ]
)
role_filter = grid_filters.selectbox(
    label = "Role",
    options = [
        "Gesture Recognition",
        "Hand Landmarker",
        "Pose Estimation (MediaPipe)"
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

#REMEMBER ABOUT TESTING WORKOUT MONITORING FROM ULTRALYTICS
if role_filter == "Gesture Recognition":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipeGestureRecognition()
        recognition_result = model.transform(opencv_image)
        grid2.subheader(role_filter)
        #grid2.write(recognition_result.gestures[0])
        if recognition_result.gestures != []:
            grid2_cols = grid2.columns(len(recognition_result.gestures[0]))
            for i in range(len(recognition_result.gestures[0])):
                grid2_cols[i].metric(
                    label = recognition_result.gestures[0][i].category_name,
                    value = "{:.2f}%".format(recognition_result.gestures[0][i].score * 100)
                )
        grid2.image(opencv_image, use_column_width = True)
elif role_filter == "Hand Landmarker":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipeHandLandmarker()
        img = model.transform(opencv_image)
        grid2.subheader(role_filter)
        grid2.image(img, use_column_width = True)
elif role_filter == "Pose Estimation (MediaPipe)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipePoseEstimation()
        img = model.transform(opencv_image)
        grid2.subheader(role_filter)
        grid2.image(img, use_column_width = True)