import streamlit as st
import json
from streamlit_extras.grid import grid
from streamlit_extras.switch_page_button import switch_page
from functions import (
    option_menu,
    image_border_radius
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

images_json = json.load(open("assets/images.json"))
second_column.latex("\\Huge{\\textbf{COELHO VISION}}")
second_column.markdown("<i><h3>Object Detection</h3></i>", unsafe_allow_html = True)
cols1 = second_column.columns(3)
for i, x in enumerate([
    "Face Detection",
    "Image Classification",
    "Object Detection"
]):
    with cols1[i]:
        image_border_radius(images_json[x], 10, 100, 25, is_html = True)
        st.caption(x)
second_column.markdown("<i><h3>Image Segmentation</h3></i>", unsafe_allow_html = True)
cols2 = second_column.columns(3)
for i, x in enumerate([
    "Image Segmentation"
]):
    with cols2[i]:
        image_border_radius(images_json[x], 10, 100, 25, is_html = True)
        st.caption(x)
second_column.markdown("<i><h3>Pose Estimation</h3></i>", unsafe_allow_html = True)
cols3 = second_column.columns(3)
for i, x in enumerate([
    "Gesture Recognition",
    "Hand Landmarker",
    "Pose Estimation"
]):
    with cols3[i]:
        image_border_radius(images_json[x], 10, 100, 25, is_html = True)
        st.caption(x)
st.divider()

with st.expander(
    label = "COELHO VISION",
    expanded = True
):
    cols1 = grid(5)
    cols2 = grid(5)
    for i in range(1, 6):
        cols1.image(f"assets/coelhovision{i:0>2}.png")
    for i in range(6, 11):
        cols2.image(f"assets/coelhovision{i:0>2}.png")
st.divider()