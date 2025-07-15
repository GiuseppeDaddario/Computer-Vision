import os
from pathlib import Path

# Settings
IMG_WIDTH = 1160
IMG_HEIGHT = 720
CLASS_ID = 0 

# Building the bounding box
def convert_bbox(x1, y1, x2, y2):
    x_center = (x1 + x2) / 2.0 / IMG_WIDTH
    y_center = (y1 + y2) / 2.0 / IMG_HEIGHT
    width = abs(x2 - x1) / IMG_WIDTH
    height = abs(y2 - y1) / IMG_HEIGHT
    return x_center, y_center, width, height

# Obtaining the measures from the image file name
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

# Processing the list of names in a .txt file
def process_list_file(list_file, label_dir):
    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        image_path = line.strip()
        filename = Path(image_path).name
        bbox = parse_filename(filename)
        if bbox is None:
            continue

        label_path = Path(label_dir) / (Path(image_path).stem + ".txt")
        os.makedirs(label_path.parent, exist_ok=True)

        with open(label_path, 'w') as out_f:
            out_f.write(f"{CLASS_ID} {' '.join(f'{x:.6f}' for x in bbox)}\n")


# (main) Processing all the .txt files
process_list_file("train.txt", "labels/train")
process_list_file("val.txt", "labels/val")
process_list_file("test.txt", "labels/test")