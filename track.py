from ultralytics import YOLO

model = YOLO('/home/ray/dev/PDI/aps/poring.v1i.yolov8/runs/detect/train5/weights/best.pt')

results = model.track(source="video.mp4", save=True)