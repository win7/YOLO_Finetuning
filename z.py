# from ultralytics import YOLOv10
from ultralytics import YOLO

# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# model = YOLOv10('runs_yolov10/custom_class_training/weights/last.pt')
# model = YOLO('runs_yolov10/custom_class_training/weights/best.pt')
model = YOLO("yolov10s.pt")

source = 'http://images.cocodataset.org/val2017/000000039769.jpg'

# Predict
# model.predict(source=source, save=True)

# Export
# model.export(format="onnx", opset=13, simplify=True)
model.export(format="onnx", opset=12, simplify=True, nms=False)
print("Inference completed.")