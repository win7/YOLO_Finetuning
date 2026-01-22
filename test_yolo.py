from ultralytics import YOLO

def main():
    # Load trained YOLOv10 model
    # model = YOLO("yolov10s.pt")  # Load a pretrained YOLOv10s model
    model = YOLO("/home/ealvarez/Project/Yolo_Test/runs_yolo11/custom_class_training/weights/best.pt")

    # Inference
    image_list = [
        "datasets/coco_franco/test/images/180demo_mp4-0004_jpg.rf.a440a0fde966176dd62862e3cdbdef36.jpg",
        "datasets/coco_franco/train/images/field2_mp4-0006_jpg.rf.a744404aa53a91fbed36b85899b8b22c.jpg",
        "datasets/coco_franco/train/images/VID-20240320-WA0001_mp4-0008_jpg.rf.64cc5548d249175d01316cdc37e4493f.jpg",
        "/home/ealvarez/Project/Yolo_Test/results/data/premium_photo.jpg",
        "/home/ealvarez/Project/Yolo_Test/results/data/panels.jpg",
        "/home/ealvarez/Project/Yolo_Test/franco/train/images/WhatsApp-Video-2024-03-29-at-9_25_47-AM_mp4-0009_jpg.rf.12ca890faa8a754d94533355d81a7455.jpg",
        "/home/ealvarez/Project/Yolo_Test/results/data/bus_.jpg"
    ]
    
    # Run inference and save annotated results
    model(
        source=image_list,
        device=0,
        save=True,                     # Save images with bounding boxes
        project="results",
        name="test_all_classes",
        exist_ok=True
    )
    print("Inference image saved in: results/inference_test/")

    # Validation metrics
    # metrics = model.val()
    # print(metrics)
    print("Inference image saved in: results/inference_test/")

    # Validation metrics
    # metrics = model.val()
    # print(metrics)

if __name__ == "__main__":
    main()