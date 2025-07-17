import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path

IMG_WIDTH = 1160
IMG_HEIGHT = 720

def load_yolo_bbox(label_path):
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 5:
                print(f"[WARN] Wrong format in label: {label_path}")
                return None
            _, x_center, y_center, width, height = map(float, parts)
            return x_center, y_center, width, height
    except Exception as e:
        print(f"[ERROR] Reading label {label_path}: {e}")
        return None

def save_image_with_bbox(img_path, label_path, output_path):
    img_path = Path(img_path)
    label_path = Path(label_path)
    output_path = Path(output_path)

    bbox = load_yolo_bbox(label_path)
    if bbox is None:
        print("[ERROR] Failed to load bbox.")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Cannot read image {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x_center, y_center, width, height = bbox
    # Denormalize
    x_center *= IMG_WIDTH
    y_center *= IMG_HEIGHT
    width *= IMG_WIDTH
    height *= IMG_HEIGHT
    x1 = x_center - width / 2
    y1 = y_center - height / 2

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x1, y1), width, height,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    ax.set_title(f"BBox from label: {label_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved image with bbox to {output_path}")

if __name__ == "__main__":
    image_dir = Path("/leonardo_scratch/large/userexternal/gdaddari/dataset/CCPD_YOLO/ccpd_challenge/images/test")
    label_dir = Path("/leonardo_scratch/large/userexternal/gdaddari/dataset/CCPD_YOLO/ccpd_challenge/labels/test")

    img_list = sorted(image_dir.glob("*.jpg"))
    if not img_list:
        print("No images found.")
    else:
        test_img_path = img_list[0]
        label_path = label_dir / (test_img_path.stem + ".txt")
        output_img_path = Path("check") / f"bbox_{test_img_path.name}"
        save_image_with_bbox(test_img_path, label_path, output_img_path)