import cv2
import numpy as np
from PIL import Image
import streamlit as st
import face_detection as fd


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        return None


def load_model():
    model = fd.build_detector(
        "RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)
    return model


def predict(model, image):
    detections = model.detect(image)[:,:4]
    for detection in detections:
        print(detection)
        x1, y1, x2, y2 = [int(_) for _ in detection]
        print(x1, y1, x2, y2)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
    return image


def main():
    st.title('Face detection using https://pypi.org/project/face-detection/')
    st.subheader("Repo cloned from https://github.com/RanFeldesh/streamlit-examples")
    model = load_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        newImage = predict(model, image)
        st.image(newImage)


if __name__ == '__main__':
    main()
