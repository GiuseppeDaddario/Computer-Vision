import sys
sys.path.append("src/YOLO/yolov5")

from train import training as YOLOv5_training
from detect import inference as YOLOv5_inference

__all__ = ['YOLOv5_training',
           'YOLOv5_inference']