from ultralytics import YOLO 
import cv2
from PIL import Image
import os

#input_file = "/home/ray/dev/PDI/aps/poring.v1i.yolov8/2023-11-28_14-32-02.mp4"

input_file = "D:/UTFPR/2023-2/pdi/projeto_final/yolo8PDI/video/pay_dun00.mp4"
model = YOLO('D:/UTFPR/2023-2/pdi/projeto_final/yolo8PDI/runs/detect/train20/weights/best.pt')

results = model.predict(source=input_file, save=True)
