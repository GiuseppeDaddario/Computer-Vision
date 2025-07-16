# Train YOLOv5s
from src import YOLOv5_training

TRAINING_PATH_YOLO = "dataset/ccpd_2019.yaml"

if __name__ == "main":
    
    YOLOv5_training(
        weights="yolov5s.pt",
        data=TRAINING_PATH_YOLO,
        epochs=300,
        batch_size=50,
        imgsz=640,
        optimizer="Adam",
        lr0=1e-3,
        lrf=1e-5,
        cos_lr=True,
        project="runs/train",
        name="lp_detection",
        cache="ram"
    )