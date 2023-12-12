from ultralytics import YOLO

model = YOLO('D:/UTFPR/2023-2/pdi/projeto_final/yolo8PDI/runs/detect/train20/weights/best.pt')

results = model.track(source="D:/UTFPR/2023-2/pdi/projeto_final/yolo8PDI/video/pay_dun00.mp4", save=True)