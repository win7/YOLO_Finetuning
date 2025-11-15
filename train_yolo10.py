from ultralytics import YOLO
import torch

# https://app.roboflow.com/ds/um4pf9QC8d?key=0OkNHSPBBp

def main():

    # ------------------------------------------------------------
    # 1. Select a pretrained YOLOv10 model
    #    (all YOLOv10 models are pretrained on COCO – 80 classes)
    # ------------------------------------------------------------
    model_path = "yolov10s.pt"  # Options: yolov10n.pt, yolov10s.pt, yolov10m.pt, etc.
    model = YOLO(model_path)

    # ------------------------------------------------------------
    # 2. Dataset YAML file
    #    It must include ALL existing classes + your new class
    # ------------------------------------------------------------
    data_yaml = "data_custom.yaml"

    # ------------------------------------------------------------
    # 3. Detect or specify number of GPUs
    # ------------------------------------------------------------
    num_gpus = torch.cuda.device_count()  # Auto-detect available GPUs

    print(f"Detected GPUs: {num_gpus}")
    if num_gpus == 0:
        print("⚠️ No GPU detected → training will run on CPU (very slow).")
    else:
        print(f"Training will run using {num_gpus} GPU(s).")

    # ------------------------------------------------------------
    # 4. Training configuration
    # ------------------------------------------------------------
    results = model.train(
        data=data_yaml,        # YAML file with all classes
        epochs=500,            # Increase if your new class has few samples (default 100)
        imgsz=640,             # Image size
        batch=64,              # Adjust depending on your GPU memory
        lr0=0.001,             # Initial learning rate
        pretrained=True,       # Use COCO pretrained weights
        optimizer="auto",      # Options: SGD, Adam, AdamW, auto
        device=1, # 0 if num_gpus > 0 else "cpu",  # Use first GPU or CPU fallback
        workers=8,             # Number of dataloader workers
        project="runs_yolov10",
        name="custom_class_training",
        verbose=True
    )

    model.export(format="onnx")
    
    print("Training completed.")
    print(results)

    """ # ------------------------------------------------------------
    # 5. Test inference after training
    # ------------------------------------------------------------
    image_path = "test.jpg"  # Replace with any test image
    predictions = model(image_path)
    predictions[0].show()  # Visualize predictions """

if __name__ == "__main__":
    main()
