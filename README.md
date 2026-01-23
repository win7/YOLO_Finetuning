
Python 3.10

Activate 
- conda activate yolo

Download coco: run script

Create filter labels (franco dataset)
- Run preprocessing.py

Merge coco and custom data

Copy images
cp -r images/train2017/. coco_franco/train/images/

Copy filter labels
cp -r franco/test/labels_filter/. datasets/coco_franco/test/labels/
cp -r franco/train/labels_filter/. datasets/coco_franco/train/labels/
cp -r franco/valid/labels_filter/. datasets/coco_franco/valid/labels/

Modify data_custom.yaml
- Add class 80 and 81

Run train
- run train_yolo.py


Export
- yolo export model=yolo11n.pt format=onnx      # export official model
- yolo export model=path/to/best.pt format=onnx # export custom trained model