import streamlit as st
import base64
import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import processors
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from urllib.request import urlretrieve
import os 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from st_pages import show_pages, Page, Section, add_indentation
from streamlit_extras.switch_page_button import switch_page

def option_menu():
    show_pages([
        Page("home.py", "Home"),
        Page("pages/object_detection.py", "Object Detection"),
        Page("pages/image_segmentation.py", "Image Segmentation"),
        Page("pages/pose_estimation.py", "Pose Estimation"),
        Page("pages/live_camera.py", "Live Camera")
    ])
    add_indentation()

def page_buttons():
    st.write(" ")
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

def image_border_radius(image_path, border_radius, width, height, page_object = None, is_html = False):
    if is_html == False:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        # Create HTML string with the image
        img_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)
    else:
        # Create HTML string with the image
        img_html = f'<img src="{image_path}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)

############################################################################################
#COMPUTER VISION
class FullFaceDetector:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            "./data/haarcascade_frontalface_default.xml")
        self.eye_detector = cv2.CascadeClassifier(
            "./data/haarcascade_eye.xml"
        )
        self.mouth_detector = cv2.CascadeClassifier(
            "./data/haarcascade_mcs_mouth.xml"
        )
        self.font_scale = 0.5
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    def transform(
            self, 
            img,
            face_scaleFactor = 1.50, 
            eye_scaleFactor = 1.30, 
            mouth_scaleFactor = 3.00,
            face_minNeighbors = 5,
            eye_minNeighbors = 7,
            mouth_minNeighbors = 10):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray_img, face_scaleFactor, face_minNeighbors)
            eyes = self.eye_detector.detectMultiScale(gray_img, eye_scaleFactor, eye_minNeighbors)
            mouths = self.mouth_detector.detectMultiScale(gray_img, mouth_scaleFactor, mouth_minNeighbors)
            for detector, color, name in [
                (faces, (255, 0, 0), "Face"), 
                (eyes, (0, 255, 0), "Eye"),
                (mouths, (0, 0, 255), "Mouth")]:
                for (x, y, w, h) in detector:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    (text_width, text_height), _ = cv2.getTextSize(name, self.font, fontScale = self.font_scale, thickness = 1)
                    # Create a filled rectangle for label background
                    label_background_start = (x, y - text_height - 8)  # Adjusted to position above the box
                    label_background_end = (x + text_width, y)
                    img = cv2.rectangle(img, label_background_start, label_background_end, color, thickness = cv2.FILLED)
                    img = cv2.putText(img, name, (x, y - 5), self.font, self.font_scale, (0,0,0), 1)
            return img
    
@st.cache_resource
def load_yolo_model_obj(model_size):
    model_name_dict = {
        "Nano": "yolov8n.pt",
        "Small": "yolov8s.pt",
        "Medium": "yolov8m.pt",
        "Large": "yolov8l.pt",
        "Extra Large": "yolov8x.pt"
    }
    model = YOLO(model_name_dict[model_size])
    return model

@st.cache_resource
def load_yolo_model_seg(model_size):
    model_name_dict = {
        "Nano": "yolov8n-seg.pt",
        "Small": "yolov8s-seg.pt",
        "Medium": "yolov8m-seg.pt",
        "Large": "yolov8l-seg.pt",
        "Extra Large": "yolov8x-seg.pt"
    }
    model = YOLO(model_name_dict[model_size])
    return model

def Hex_to_RGB(ip):
    return tuple(int(ip[i:i+2],16) for i in (0, 2, 4))

class MediaPipeObjectDetection:
    def __init__(self):
        pass
    def visualize(self, image, detection_result, text_color):
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, text_color, 3)   
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        FONT_SIZE, text_color, FONT_THICKNESS)    
        return image
    def transform(self, original_image, text_color):
        #Create an ObjectDetector object
        if "efficientdet.tflite" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
            filename = "efficientdet.tflite"
            urlretrieve(url, filename)
        base_options = python.BaseOptions(
            model_asset_path = 'efficientdet.tflite')
        options = vision.ObjectDetectorOptions(
            base_options = base_options,
            score_threshold = 0.5)
        detector = vision.ObjectDetector.create_from_options(options)
        #Load the input image
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        #Detect objects in the input image
        detection_result = detector.detect(image)
        #Process the detection result
        image_copy = np.copy(image.numpy_view())
        annotated_image = self.visualize(image_copy, detection_result, text_color)
        return annotated_image


class MediaPipeImageClassification:
    def __init__(self):
        pass
    def transform(self, original_image):
        if "classifier.tflite" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
            filename = "classifier.tflite"
            urlretrieve(url, filename)
        #Create an ImageClassifier object
        base_options = python.BaseOptions(
            model_asset_path = 'classifier.tflite')
        options = vision.ImageClassifierOptions(
            base_options = base_options, 
            max_results = 4)
        classifier = vision.ImageClassifier.create_from_options(options)
        #Load the input image
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        #Classify the input image
        classification_result = classifier.classify(image)
        #Process the classification result
        return classification_result


class MediaPipeImageSegmentation:
    def __init__(self):
        pass
    def transform(self, original_image, mask_color, background_color):
        if "deeplabv3.tflite" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
            filename = "deeplabv3.tflite"
            urlretrieve(url, filename)
        # Create the options that will be used for ImageSegmenter
        base_options = python.BaseOptions(
            model_asset_path = 'deeplabv3.tflite')
        options = vision.ImageSegmenterOptions(
            base_options = base_options,
            output_category_mask = True)
        # Create the image segmenter
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            # Create the MediaPipe image file that will be segmented
            image = mp.Image(
                image_format = mp.ImageFormat.SRGB,
                data = original_image
            )
            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask
            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = Hex_to_RGB(mask_color[1:])
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = Hex_to_RGB(background_color[1:])
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
        return output_image


class MediaPipeGestureRecognition:
    def __init__(self):
        pass
    def transform(self, original_image):
        if "gesture_recognizer.task" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            filename = "gesture_recognizer.task"
            urlretrieve(url, filename)
        #Create an GestureRecognizer object
        base_options = python.BaseOptions(
            model_asset_path = 'gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(
            base_options = base_options)
        recognizer = vision.GestureRecognizer.create_from_options(options)
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        recognition_result = recognizer.recognize(image)
        return recognition_result
    

class MediaPipeHandLandmarker:
    def __init__(self):
        pass
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]   
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
              annotated_image,
              hand_landmarks_proto,
              solutions.hands.HAND_CONNECTIONS,
              solutions.drawing_styles.get_default_hand_landmarks_style(),
              solutions.drawing_styles.get_default_hand_connections_style())    
            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN  
            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)  
        return annotated_image
    def transform(self, original_image):
        if "hand_landmarker.task" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            filename = "hand_landmarker.task"
            urlretrieve(url, filename)
        base_options = python.BaseOptions(
            model_asset_path = 'hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands = 2)
        detector = vision.HandLandmarker.create_from_options(options)
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        detection_result = detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        return annotated_image


class MediaPipeFaceDetector:
    def __init__(self):
        pass
    def _normalized_to_pixel_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value)) 
        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    def visualize(self, image, detection_result):
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red
        annotated_image = image.copy()
        height, width, _ = image.shape  
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3) 
            # Draw keypoints
            for keypoint in detection.keypoints:
              keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                             width, height)
              color, thickness, radius = (0, 255, 0), 2, 2
              cv2.circle(annotated_image, keypoint_px, thickness, color, radius)  
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)    
        return annotated_image
    def transform(self, original_image):
        if "detector.tflite" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            filename = "detector.tflite"
            urlretrieve(url, filename)
        base_options = python.BaseOptions(model_asset_path='detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        detection_result = detector.detect(image)
        image_copy = np.copy(image.numpy_view())
        annotated_image = self.visualize(image_copy, detection_result)
        return annotated_image
    

class MediaPipePoseEstimation:
    def __init__(self):
        pass
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
              annotated_image,
              pose_landmarks_proto,
              solutions.pose.POSE_CONNECTIONS,
              solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image
    def transform(self, original_image):
        if "pose_landmarker.task" not in os.listdir():
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            filename = "pose_landmarker.task"
            urlretrieve(url, filename)
        base_options = python.BaseOptions(
            model_asset_path = 'pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options = base_options,
            output_segmentation_masks = True)
        detector = vision.PoseLandmarker.create_from_options(options)
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        detection_result = detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        return annotated_image