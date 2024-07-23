import streamlit as st
import json
from streamlit_extras.grid import grid
from streamlit_extras.switch_page_button import switch_page
from functions import (
    option_menu,
    image_border_radius,
    image_carousel
)

st.set_page_config(
    page_title = "COELHO VISION",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

layout = grid([1, 0.2, 2], vertical_align = True)
first_column = layout.container()
layout.container()
second_column = layout.container()
image_border_radius("assets/coelho_vision_logo.png", 20, 100, 100, first_column)
first_column.caption("Author: Rafael Silva Coelho")

OBJECT_DETECTION = first_column.button(
    label = "$$\\textbf{Object Detection}$$",
    use_container_width = True)
IMAGE_SEGMENTATION = first_column.button(
    label = "$$\\textbf{Image Segmentation}$$",
    use_container_width = True)
POSE_ESTIMATION = first_column.button(
    label = "$$\\textbf{Pose Estimation}$$",
    use_container_width = True)
LIVE_CAMERA = first_column.button(
    label = "$$\\textbf{Live Camera}$$",
    use_container_width = True)
ABOUT_US = first_column.button(
    label = "$$\\textbf{About Us}$$",
    use_container_width = True)
if OBJECT_DETECTION:
    switch_page("object detection")
if IMAGE_SEGMENTATION:
    switch_page("image segmentation")
if POSE_ESTIMATION:
    switch_page("pose estimation")
if LIVE_CAMERA:
    switch_page("live camera")
if ABOUT_US:
    switch_page("about")

with first_column:
    image_carousel([
        f"assets/coelhovision{x:0>2}.png" for x in range(1, 11)
    ], [])

images_json = json.load(open("assets/images.json"))
second_column.latex("\\Huge{\\textbf{COELHO VISION}}")
second_column.divider()
second_column_cols = second_column.columns(3)
with second_column_cols[0]:
    st.markdown("<i><h3>Object Detection</h3></i>", unsafe_allow_html = True)
    image_carousel([
        "assets/home_fullfacedetector.png",
        "assets/home_objecttracking.jpg"
    ], [
        images_json[x] for x in [
            "Face Detection",
            "Image Classification",
            "Object Detection"
        ]
    ])
with second_column_cols[1]:
    st.markdown("<i><h3>Image Segmentation</h3></i>", unsafe_allow_html = True)
    image_carousel([
        "assets/home_depthestimation.png",
        "assets/home_semanticsegmentation.png"
    ], [
        images_json[x] for x in [
            "Image Segmentation"
        ]
    ])
with second_column_cols[2]:
    st.markdown("<i><h3>Pose Estimation</h3></i>", unsafe_allow_html = True)
    image_carousel([], [
        images_json[x] for x in [
            "Gesture Recognition",
            "Hand Landmarker",
            "Pose Estimation"
        ]
    ])
st.divider()