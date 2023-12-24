import streamlit as st
from streamlit_extras.grid import grid
import cv2
import av
from streamlit_webrtc import webrtc_streamer
from functions import (
    option_menu,
    page_buttons,
    image_border_radius
)
from functions import (
    MediaPipeFaceDetector
)

st.set_page_config(
    page_title = "COELHO VISION | Live Camera",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

grid_title = grid([5, 1], vertical_align = True)
container1 = grid_title.container()
container1.title("$$\\large{\\textbf{COELHO VISION | Live Camera}}$$")
container1.caption("Author: Rafael Silva Coelho")

page_buttons()

st.divider()
image_border_radius("./assets/coelho_vision_logo.png", 20, 100, 100, grid_title)

grid_filters = grid([1, 3], vertical_align = True)
grid1 = grid_filters.container()
grid2 = grid_filters.container()

task_filter = grid1.selectbox(
    label = "Task type",
    options = [
        "Simple task"
    ]
)
role_filter = grid1.selectbox(
    label = "Role",
    options = [
        "Face Detector (MediaPipe)"
    ]
)

if role_filter == "Face Detector (MediaPipe)":
    model = MediaPipeFaceDetector()

def video_frame_callback(frame):
    image = frame.to_ndarray(format = "bgr24")
    final_image = model.transform(image)
    return av.VideoFrame.from_ndarray(final_image, format="bgr24")

with grid2:
    webrtc_streamer(
        key = "live_camera",
        video_frame_callback = video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints = {"video": True, "audio": False}
    )