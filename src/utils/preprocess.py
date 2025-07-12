from PIL import Image




def crop_plate(image_path, bbox):
    image = Image.open(image_path)
    (x1, y1), (x2, y2) = bbox
    plate_crop = image.crop((x1, y1, x2, y2))  # (left, upper, right, lower)
    return plate_crop
