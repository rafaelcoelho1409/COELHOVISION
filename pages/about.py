import datetime as dt
import streamlit as st
from streamlit_extras.grid import grid
from streamlit_card import card
from streamlit_extras.switch_page_button import switch_page
from functions import option_menu, image_border_radius

st.set_page_config(
    page_title = "COELHO VISION - About Us",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

option_menu()

st.title("$$\\large{\\textbf{About Us}}$$")

cols_ = st.columns(5)
with cols_[0]:
    HOME = st.button(
        label = "$$\\textbf{Home}$$",
        use_container_width = True)
with cols_[1]:
    OBJECT_DETECTION = st.button(
        label = "$$\\textbf{Object Detection}$$",
        use_container_width = True)
with cols_[2]:
    IMAGE_SEGMENTATION = st.button(
        label = "$$\\textbf{Image Segmentation}$$",
        use_container_width = True)
with cols_[3]:
    POSE_ESTIMATION = st.button(
        label = "$$\\textbf{Pose Estimation}$$",
        use_container_width = True)
with cols_[4]:
    LIVE_CAMERA = st.button(
        label = "$$\\textbf{Live Camera}$$",
        use_container_width = True)
if HOME:
    switch_page("home")
if OBJECT_DETECTION:
    switch_page("object detection")
if IMAGE_SEGMENTATION:
    switch_page("image segmentation")
if POSE_ESTIMATION:
    switch_page("pose estimation")
if LIVE_CAMERA:
    switch_page("live camera")

with st.expander(
    label = "Author",
    expanded = True
):
    st.write("$$\\underline{\\Large{\\textbf{Author}}}$$")
    grid1 = grid([1, 0.1, 4], vertical_align = True)
    image_border_radius("assets/rafael_coelho_1.jpeg", 20, 80, 80, grid1)
    grid1.container()
    container1 = grid1.container()
    container1.markdown(f"""<div style='font-size:25px'>
    Rafael Coelho is a Brazilian Mathematics student 
    who is passionated for Data Science and Artificial Intelligence
    and works in both areas for over {str(dt.datetime.now().year - 2020)} years, with solid knowledge in
    technologic areas such as Machine Learning, Deep Learning, Data Science,
    Computer Vision, Reinforcement Learning, NLP and others.<br><br>
    Recently, he worked in one of the Big Four companies for over a year.</div>
    """, unsafe_allow_html = True)
    container1.divider()
    buttons = container1.columns(3)
    buttons[0].link_button(
        "Portfolio",
        "https://rafaelcoelho.streamlit.app/",
        type = "primary",
        use_container_width = True
    )
    buttons[1].link_button(
        "LinkedIn",
        "https://www.linkedin.com/in/rafaelcoelho1409/",
        type = "primary",
        use_container_width = True
    )
    buttons[2].link_button(
        "GitHub",
        "https://github.com/rafaelcoelho1409/",
        type = "primary",
        use_container_width = True
    )
####################################

