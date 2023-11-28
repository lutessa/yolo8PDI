from ultralytics import YOLO 
import cv2
from PIL import Image
import os

input_folder = "/home/ray/dev/PDI/aps/poring.v1i.yolov8/test/images/"
output_folder = "/home/ray/dev/PDI/aps/poring.v1i.yolov8/results/"

model = YOLO('/home/ray/dev/PDI/aps/poring.v1i.yolov8/runs/detect/train5/weights/best.pt')


image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]


for image_file in image_files:

    input_path = os.path.join(input_folder, image_file)
    

    im2 = cv2.imread(input_path)
    

    #results = model.predict(source=im2, save=True, save_txt=True)
    results = model.predict(source=im2, project=output_folder, name=image_file, save=True)

    # output_path = os.path.join(output_folder, image_file)
    # results.save(output_path)