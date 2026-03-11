from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # base model

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"
)
