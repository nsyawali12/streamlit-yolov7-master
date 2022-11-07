import streamlit as st
import cv2
import torch
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile
import pandas as pd
from PIL import ImageColor


st.title('Vibrio Counter Prediction')
sample_img = cv2.imread('sample.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
st.sidebar.title('Settings')

# path to model
path_model_file = 'last.pt'

# Class txt
path_to_class_txt = 'class_vibrio.txt'

# path_to_class_txt = 'class_vibrio.txt'

if path_to_class_txt is not None:

    with open(path_to_class_txt, "r") as f:
        class_labels = f.readlines()
    # options = st.sidebar.radio(
    #     'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)
    
    options = 'Image'

    # gpu_option = st.sidebar.radio(
    #     'PU Options:', ('CPU', 'GPU'))

    # if not torch.cuda.is_available():
    #     st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
    # else:
    #     st.sidebar.success(
    #         'GPU is Available on this Device, Choose GPU for the best performance',
    #         icon="✅"
    #     )
    
    gpu_option = 'CPU'

    # Confidence
    # confidence = st.sidebar.slider(
    #     'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)
    
    confidence = 0.25

    # Draw thickness
    # draw_thick = st.sidebar.slider(
    #     'Draw Thickness:', min_value=1,
    #     max_value=20, value=3
    # )
    
    # Draw thickness
    draw_thick = 3

    # Color picker
    
    color_picke = st.sidebar.color_picker('Draw Color:', '#ff0003')
    # color_pickle = '#ff0003'
    
    
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    
    color = [color_rgb_list[1], color_rgb_list[2], color_rgb_list[0]]
    
    

    # Image
    if options == 'Image':
        upload_img_file = st.sidebar.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            pred = st.checkbox('Predict Using YOLOv7')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')

            if pred:
                if gpu_option == 'CPU':
                    model = custom(path_or_model=path_model_file)
                if gpu_option == 'GPU':
                    model = custom(path_or_model=path_model_file, gpu=True)
                bbox_list = []
                results = model(img)
                # Bounding Box
                box = results.pandas().xyxy[0]
                class_list = box['class'].to_list()

                # read class.txt
                # bytes_data = path_to_class_txt

                for i in box.index:
                    xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                        int(box['ymax'][i]), box['confidence'][i]
                    if conf > confidence:
                        bbox_list.append([xmin, ymin, xmax, ymax])
                if len(bbox_list) != 0:
                    for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(bbox, img, label=class_labels[id],
                                     color=color, line_thickness=draw_thick)
                
                boxes = pd.DataFrame(box)
                group_box = boxes.groupby(by=["name"]).count()["class"]
                st.dataframe(group_box)
                conf_box = boxes.groupby(by=["name"]).mean()["confidence"]
                st.dataframe(conf_box)
                FRAME_WINDOW.image(img, channels='BGR')