# Train YOLOv5s
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src import YOLOv5_training

TRAINING_PATH_YOLO = "dataset/ccpd_2019.yaml"

if __name__=="__main__":
    
    YOLOv5_training(
        weights="yolov5s.pt",
        data=TRAINING_PATH_YOLO,
        epochs=3,
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