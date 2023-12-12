from ultralytics import YOLO 
import cv2
from PIL import Image
import torch


# setup GPU
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)
###

torch.cuda.empty_cache()


print("Device: ", device)

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='data.yaml', epochs=100, batch=12)

# Evaluate the model's performance on the validation set
results = model.val()