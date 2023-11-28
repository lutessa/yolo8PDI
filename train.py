from ultralytics import YOLO 
import cv2
from PIL import Image

# # Create a new YOLO model from scratch
# #model = YOLO('yolov8n.yaml')
# model = YOLO('yolov8n-seg.yaml')
# # Load a pretrained YOLO model (recommended for training)
# #model = YOLO('yolov8n.pt')
# model = YOLO('yolov8n-seg.pt') 
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='data.yaml', epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('C:/Users/L04578/Documents/Mexilhao/Fotos_Microscopio_2023.10.11/2023.11.10_10.10.52_image_154.png')


# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray

print("--------------TESTING--------------------")
im2 = cv2.imread('/home/ray/dev/PDI/aps/poring.v1i.yolov8/test/images/img_1_jpg.rf.fe074ed625284ea25318b7b62e272ac5.jpg')
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
#results = model.predict(source=[ im2])
# Export the model to ONNX format
# success = model.export(format='onnx')