from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.predict("data/test.mp4", device="mps", save=True)