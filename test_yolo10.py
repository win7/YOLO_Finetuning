from ultralytics import YOLO

def main():
    # Load trained YOLOv10 model
    model = YOLO("runs_yolov10/custom_class_training3/weights/best.pt")
    model = YOLO("yolov10s.pt")  # Load a pretrained YOLOv10s model

    # Inference
    img = "data/test/images/180demo_mp4-0004_jpg.rf.a440a0fde966176dd62862e3cdbdef36.jpg"
    img = "data/train/images/field2_mp4-0006_jpg.rf.a744404aa53a91fbed36b85899b8b22c.jpg"
    img = "data/train/images/VID-20240320-WA0001_mp4-0008_jpg.rf.64cc5548d249175d01316cdc37e4493f.jpg"
    img = "results/premium_photo-1683133462626-c3bb80a407db.jpg"
    
    # Run inference and save annotated results
    model(
        img,
        device=1,
        save=True,                     # Save images with bounding boxes
        project="results",
        name="test_all_classes",
        exist_ok=True
    )
    print("Inference image saved in: results/inference_test/")

    # Validation metrics
    # metrics = model.val()
    # print(metrics)

if __name__ == "__main__":
    main()