# Building the .txt files, one for each image, with meta-data information in YOLO format
# TODO: Needs to be revisited for Leonardo

import os
os.environ['WANDB_MODE'] = 'disabled'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import shutil

IMG_WIDTH = 1160
IMG_HEIGHT = 720
CLASS_ID = 0 

######################## UTILS FOR YOLOV5 ########################
# [Data Pre-processing] Building the bounding box in YOLO format
def convert_bbox(x1, y1, x2, y2):
    x_center = (x1 + x2) / 2.0 / IMG_WIDTH
    y_center = (y1 + y2) / 2.0 / IMG_HEIGHT
    width = abs(x2 - x1) / IMG_WIDTH
    height = abs(y2 - y1) / IMG_HEIGHT
    return x_center, y_center, width, height

# [Data Pre-processing] Extracting meta-data from the image file name
def parse_filename(fname):
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

# [Data Pre-processing] Processing the list of names in a .txt file
def prepare_yolo_dataset(src, dest_root="dataset", split="train", debug=False):
    images_src = f"dataset/CCPD2019/{src}"
    images_src = Path(images_src)
    dest_root = f"dataset/{dest_root}/{src}"
    images_dest = Path(dest_root) / "images" / split
    labels_dest = Path(dest_root) / "labels" / split

    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

    image_files = list(images_src.glob("*.jpg"))[:5] if debug else list(images_src.glob("*.jpg")) 

    for img_path in tqdm(image_files, desc=f"Processing {split} set"):
        bbox = parse_filename(img_path.name)
        if bbox is None:
            continue

        # Copy image
        dest_img = images_dest / img_path.name
        shutil.copy(img_path, dest_img)

        # Build label
        label_path = labels_dest / (img_path.stem + ".txt")
        with open(label_path, 'w') as f:
            f.write(f"{CLASS_ID} {' '.join(f'{x:.6f}' for x in bbox)}\n")

    print(f"Saved to {images_dest} and {labels_dest}")

##################################################################

if __name__=="main":
    subsets = [
            "ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db",
            "ccpd_fn", "ccpd_np", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ]
    splits = ["train", "val", "test"]
    dest_root = "ccpd2019_yolo"

    tasks = [(subset, dest_root, split) for subset in subsets for split in splits]

        # Usa un numero di processi pari ai core disponibili o meno se vuoi evitare overload
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(prepare_yolo_dataset, tasks)
