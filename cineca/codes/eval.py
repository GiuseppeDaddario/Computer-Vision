from src import YOLOv5_inference

test_dir = "ccpd_challenge"
TEST_PATH_YOLOV5 = f"dataset/CCPD_YOLO/{test_dir}/images/test" 

if __name__ == "main":

    YOLOv5_inference(
        weights="yolov5/runs/train/exp/weights/best.pt",
        source=TEST_PATH_YOLOV5,
        imgsz=640,
        device="cuda:0",
        project="runs/detect",
        name="lp_test",
        exist_ok=True
    )