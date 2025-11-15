# yolo export model=yolo11n.pt format=onnx      # export official model
# yolo export model=path/to/best.pt format=onnx # export custom trained model


1. Download coco: run script
2. Merge coco and custom data

cp -r images/train2017/. coco_franco/train/images/

cp -r coco/labels/train2017/. coco_franco/train/labels/

3. Train
