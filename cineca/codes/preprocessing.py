# Building the .txt files, one for each image, with meta-data information in YOLO format

import os
os.environ['WANDB_MODE'] = 'disabled'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import shutil
import random

IMG_WIDTH = 1160
IMG_HEIGHT = 720
CLASS_ID = 0 

######################## UTILS FOR YOLOV5 ########################
def convert_bbox(x1, y1, x2, y2):
    """
    Converts bbox coordinates in pixels (normalized), following the YOLO format.
    """
    x_center = (x1 + x2) / 2.0 / IMG_WIDTH
    y_center = (y1 + y2) / 2.0 / IMG_HEIGHT
    width = abs(x2 - x1) / IMG_WIDTH
    height = abs(y2 - y1) / IMG_HEIGHT
    return x_center, y_center, width, height

def parse_filename(fname):
    """
    Extracts bbox coordinates from the image file name and converts them in YOLO format
    """
    parts = fname.split('-')
    if len(parts) < 4:
        return None
    bbox_part = parts[2]
    try:
        x1y1, x2y2 = bbox_part.split('_')
        x1, y1 = map(int, x1y1.split('&'))
        x2, y2 = map(int, x2y2.split('&'))
        return convert_bbox(x1, y1, x2, y2)
    except:
        return None

def process_single_image(args):
    img_path, images_dest, labels_dest = args
    bbox = parse_filename(img_path.name)
    if bbox is None:
        return

    shutil.copy(img_path, images_dest / img_path.name)
    label_path = labels_dest / (img_path.stem + ".txt")
    with open(label_path, 'w') as f:
        f.write(f"{CLASS_ID} {' '.join(f'{x:.6f}' for x in bbox)}\n")


def process_images(images, images_src, dest_root, split):
    images_dest = Path(dest_root) / "images" / split
    labels_dest = Path(dest_root) / "labels" / split
    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

    args_list = [(img_path, images_dest, labels_dest) for img_path in images]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_image, args_list), total=len(args_list), desc=f"Processing {split} set"))

    print(f"{split} set saved to {images_dest} and {labels_dest}")

def prepare_ccpd_base(src_root, dest_root="CCPD_YOLO", split_ratio=0.8, seed=42):
    """
    Builds the training subdataset 'ccpd_base'.
    """
    src = "ccpd_base"
    images_src = Path(os.path.join(src_root, src))
    image_files = list(images_src.glob("*.jpg"))
    print(f"[INFO] Found {len(image_files)} images in {images_src}")
    random.seed(seed)
    random.shuffle(image_files)

    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    process_images(train_files, images_src, f"{dest_root}/{src}", "train")
    process_images(val_files, images_src, f"{dest_root}/{src}", "val")

def prepare_other_subset(src_root, subset, dest_root="CCPD_YOLO"):
    """
    Builds the other subdatasets for the testing phase (individually).
    """
    images_src = Path(os.path.join(src_root, subset))
    image_files = list(images_src.glob("*.jpg"))
    process_images(image_files, images_src, f"{dest_root}/{subset}", "test")

##################################################################

if __name__=="__main__":
    print("Starting preprocessing...")
    scratch_dir = os.environ.get("SCRATCH", "/leonardo_scratch/large/userexternal/gdaddari")
    SRC_ROOT = os.path.join(scratch_dir, "dataset", "CCPD2019")
    base_dest = os.path.join(scratch_dir, "dataset", "CCPD_YOLO")

    other_subsets = [
        "ccpd_blur", "ccpd_challenge", "ccpd_db",
        "ccpd_fn", "ccpd_np", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ]

    print("Training subset...")
    prepare_ccpd_base(SRC_ROOT, dest_root=base_dest)

    print("Other subsets...")
    for subset in other_subsets:
        print(f"Processing {subset}...")
        prepare_other_subset(SRC_ROOT, subset, base_dest)