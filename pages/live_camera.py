import streamlit as st
from streamlit_extras.grid import grid
import cv2
import av
from streamlit_webrtc import webrtc_streamer
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
    MediaPipeFaceDetector,
    ImageSegmentationYOLO,
    MediaPipeImageSegmentation,
    MediaPipeHandLandmarker,
    MediaPipePoseEstimation
)

st.set_page_config(
    page_title = "COELHO VISION | Live Camera",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

grid_title = grid([5, 1], vertical_align = True)
container1 = grid_title.container()
container1.title("$$\\large{\\textbf{COELHO VISION}}$$")
container1.write("$$\\Huge{\\textit{\\textbf{Live Camera}}}$$")
container1.caption("Author: Rafael Silva Coelho")

page_buttons()

st.divider()
image_border_radius("./assets/coelho_vision_logo.png", 20, 100, 100, grid_title)

def video_frame_callback(frame):
    image = frame.to_ndarray(format = "bgr24")
    final_image = model.transform(image)
    return av.VideoFrame.from_ndarray(final_image, format = "bgr24")

grid_filters = grid([1, 2], vertical_align = True)
grid1 = grid_filters.container()
grid2 = grid_filters.container()

task_filter = grid1.selectbox(
    label = "Task type",
    options = [
        "Object Detection",
        "Image Segmentation",
        "Pose Estimation"
    ]
)
if task_filter == "Object Detection":
    role_filter = grid1.selectbox(
        label = "Role",
        options = [
        "Full Face Detection",
        "Object Detection (YOLOv8)",
        "Object Detection (MediaPipe)",
        "Face Detector (MediaPipe)"
    ]
    )
    if role_filter == "Full Face Detection":
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
        model = FullFaceDetector()
        model.face_scaleFactor = face_scaleFactor
        model.eye_scaleFactor = eye_scaleFactor
        model.mouth_scaleFactor = mouth_scaleFactor
        model.face_minNeighbors = face_minNeighbors
        model.eye_minNeighbors = eye_minNeighbors
        model.mouth_minNeighbors = mouth_minNeighbors
    elif role_filter == "Object Detection (YOLOv8)":
        model_size = grid1.selectbox(
            label = "YOLO Model Size",
            options = [
                "Nano",
                "Small",
                "Medium",
                "Large",
                "Extra Large"
            ]
        )
        model = ObjectDetectionYOLO()
        model.model_size = model_size
    elif role_filter == "Object Detection (MediaPipe)":
        text_color_picker = grid1.color_picker(
            label = "Set a color to bounding boxes and text"
        )
        text_color = Hex_to_RGB(text_color_picker[1:])
        model = MediaPipeObjectDetection()
        model.text_color = text_color
    elif role_filter == "Face Detector (MediaPipe)":
        model = MediaPipeFaceDetector()
elif task_filter == "Image Segmentation":
    role_filter = grid1.selectbox(
        label = "Role",
        options = [
        "Image Segmentation (YOLOv8)",
        "Image Segmentation (MediaPipe)"
    ]
    )
    if role_filter == "Image Segmentation (YOLOv8)":
        model_size = grid1.selectbox(
            label = "YOLO Model Size",
            options = [
                "Nano",
                "Small",
                "Medium",
                "Large",
                "Extra Large"
            ]
        )
        model = ImageSegmentationYOLO()
        model.model_size = model_size
    elif role_filter == "Image Segmentation (MediaPipe)":
        background_color_picker = grid1.color_picker(
            label = "Set a color to background",
            value = "#808080"
        )
        mask_color_picker = grid1.color_picker(
            label = "Set a color to mask",
            value = "#0000ff"
        )
        model = MediaPipeImageSegmentation()
        model.background_color = background_color_picker
        model.mask_color = mask_color_picker
elif task_filter == "Pose Estimation":
    role_filter = grid1.selectbox(
        label = "Role",
        options = [
        "Hand Landmarker",
        "Pose Estimation (MediaPipe)"
    ]
    )
    if role_filter == "Hand Landmarker":
        model = MediaPipeHandLandmarker()
    elif role_filter == "Pose Estimation (MediaPipe)":
        model = MediaPipePoseEstimation()



with grid2:
    ctx = webrtc_streamer(
        key = "live_camera",
        video_frame_callback = video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints = {"video": True, "audio": False}
    )