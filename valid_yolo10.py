from ultralytics import YOLO

# Load trained YOLOv10 model
model = YOLO("runs_yolov10/custom_class_training/weights/best.pt")

print(model.names)
print("Número total de clases:", len(model.names))

model(
    device=1
)
metrics = model.val()
print(metrics)