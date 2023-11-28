from ultralytics import YOLO 
import cv2
from PIL import Image
import os

#input_file = "/home/ray/dev/PDI/aps/poring.v1i.yolov8/2023-11-28_14-32-02.mp4"

input_file = "/home/ray/dev/PDI/aps/poring.v1i.yolov8/video.mp4"
model = YOLO('/home/ray/dev/PDI/aps/poring.v1i.yolov8/runs/detect/train5/weights/best.pt')

results = model.predict(source=input_file, save=True)
