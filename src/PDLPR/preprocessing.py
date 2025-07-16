from PIL import Image
import os
from torch.cuda.amp import autocast, GradScaler

def parse_box_from_filename(filename):
    # Esempio: filename = "074-153_423-245&374_263&409-..."
    parts = filename.split('-')
    box_str = parts[2]  # "245&374_263&409"
    (x1y1, x2y2) = box_str.split('_')
    x1, y1 = map(int, x1y1.split('&'))
    x2, y2 = map(int, x2y2.split('&'))
    return x1, y1, x2, y2

def crop_plate(img_path):
    img = Image.open(img_path).convert("RGB")
    filename = os.path.basename(img_path)
    x1, y1, x2, y2 = parse_box_from_filename(filename)
    # Assicurati che la bbox sia valida
    left, top = min(x1, x2), min(y1, y2)
    right, bottom = max(x1, x2), max(y1, y2)
    return img.crop((left, top, right, bottom))  # crop(topleft_x, y, bottomright_x, y)