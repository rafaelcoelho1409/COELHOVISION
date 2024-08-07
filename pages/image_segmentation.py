import streamlit as st
import numpy as np
import cv2
from ultralytics.utils.plotting import Annotator, colors
from streamlit_extras.grid import grid
from functions import (
    option_menu,
    page_buttons,
    image_border_radius,
    Hex_to_RGB
)
from functions import (
    ImageSegmentationYOLO,
    MediaPipeImageSegmentation,
    OpenVINODepthEstimation,
    OpenVINOImageSegmentation
)

st.set_page_config(
    page_title = "COELHO VISION | Image Segmentation",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

grid_title = grid([5, 1], vertical_align = True)
container1 = grid_title.container()
container1.title("$$\\large{\\textbf{COELHO VISION}}$$")
container1.write("$$\\Huge{\\textit{\\textbf{Image Segmentation}}}$$")
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
        "Image Segmentation (YOLOv8)",
        "Image Segmentation (MediaPipe)",
        "Depth Estimation (OpenVINO)",
        "Image Segmentation (OpenVINO)"
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

if role_filter == "Image Segmentation (YOLOv8)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = ImageSegmentationYOLO()
        img = model.transform(opencv_image)
        grid2.header(role_filter)
        grid2.image(img, use_column_width = True)
elif role_filter == "Image Segmentation (MediaPipe)":
    background_color_picker = grid1.color_picker(
        label = "Set a color to background",
        value = "#808080"
    )
    mask_color_picker = grid1.color_picker(
        label = "Set a color to mask",
        value = "#0000ff"
    )
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = MediaPipeImageSegmentation()
        model.mask_color = mask_color_picker
        model.background_color = background_color_picker
        img = model.transform(opencv_image)
        grid1.image(opencv_image, use_column_width = True)
        grid2.subheader(role_filter)
        grid2.image(img, use_column_width = True)
elif role_filter == "Depth Estimation (OpenVINO)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = OpenVINODepthEstimation()
        img = model.transform(opencv_image)
        grid2.header(role_filter)
        grid2.image(img, use_column_width = True)
elif role_filter == "Image Segmentation (OpenVINO)":
    if (
        mode_filter == "Image" and uploaded_image is not None) or (
        mode_filter == "Camera" and cam_image is not None    
        ):
        model = OpenVINOImageSegmentation()
        img = model.transform(opencv_image)
        grid2.header(role_filter)
        grid2.image(img, use_column_width = True)