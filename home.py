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
    switch_page("about us")

with first_column:
    image_carousel([
        f"assets/coelhovision{x:0>2}.png" for x in range(1, 11)
    ], [])

images_json = json.load(open("assets/images.json"))
second_column.latex("\\Huge{\\textbf{COELHO VISION}}")
#second_column.divider()
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
    for x in [
        "Full Face Detection",
        "Image Classification",
        "Object Detection",
        "Face Detector",
        "Object Tracking"
    ]:
        st.markdown(f"- **{x}**")
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
    for x in [
        "Image Segmentation",
        "Depth Estimation",
    ]:
        st.markdown(f"- **{x}**")
with second_column_cols[2]:
    st.markdown("<i><h3>Pose Estimation</h3></i>", unsafe_allow_html = True)
    image_carousel([], [
        images_json[x] for x in [
            "Gesture Recognition",
            "Hand Landmarker",
            "Pose Estimation"
        ]
    ])
    for x in [
        "Gesture Recognition",
        "Hand Landmarker",
        "Pose Estimation",
    ]:
        st.markdown(f"- **{x}**")
second_column.divider()
second_column.latex("\\huge{\\textbf{COELHO VISION - Software Version}}")
second_column_cols2 = second_column.columns(2)
with second_column_cols2[0]:
    image_carousel([
        f"assets/software{x}.png" for x in range(1, 5)
    ], [])
with second_column_cols2[1]:
    st.markdown(f"""<div style='font-size:15px'>
    COELHO VISION is an advanced Computer Vision software offering features 
    like Object Detection, Object Tracking, Image Classification, Image Segmentation, 
    Depth Estimation, and Pose Estimation. The Windows version provides a significant 
    performance boost, minimizing latency and ensuring a smoother, faster experience 
    compared to the web version.<br><br>
    Optimize your computer vision projects with the powerful and reliable Windows 
    version of COELHO VISION. Click the link below to download and experience the difference!
    <br><br></div>
    """, unsafe_allow_html = True)
second_column.link_button(
    label = "Download COELHO VISION software (Windows version)",
    url = "https://drive.google.com/file/d/1RyA8GgkJQDJuv6afNIKWVq-fuyhbtSNY/view?usp=drive_link",
    type = "primary",
    use_container_width = True
)
second_column.markdown("""
**Disclaimer:** for Linux and Mac versions, there is no software version.
You must clone the source repository and run it using Python 
([GitHub repository](https://github.com/rafaelcoelho1409/COELHOVISION))""")
st.divider()