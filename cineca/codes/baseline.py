import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# Disattiva wandb
os.environ['WANDB_MODE'] = 'disabled'

# GLOBALS
train_folder = "/leonardo_scratch/large/userexternal/gdaddari/dataset/CCPD_YOLO/ccpd_base/images/train"
val_folder = "/leonardo_scratch/large/userexternal/gdaddari/dataset/CCPD_YOLO/ccpd_base/images/val"
test_subsets = [
        "ccpd_blur", "ccpd_challenge", "ccpd_db",
        "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ]
save_path = "models/baseline"
W_orig, H_orig = 720, 1160
W_resize, H_resize = 224, 224
x_scale, y_scale = W_resize / W_orig, H_resize / H_orig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
N_EPOCHS = 3
BATCH_SIZE = 256

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
ads = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]

# TRANSFORMS
transform_detection = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
transform_recognition = transforms.Compose([transforms.Resize((64, 128)), transforms.ToTensor()])


# PARSER
def parse_label(filename):
    parts = filename.split('-')
    box_str = parts[2]
    x1y1, x2y2 = box_str.split('_')
    x1, y1 = map(int, x1y1.split('&'))
    x2, y2 = map(int, x2y2.split('&'))
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

def compute_iou(preds, gts):
   
    intersection_x1 = np.maximum(preds[:, 0], gts[:, 0])
    intersection_y1 = np.maximum(preds[:, 1], gts[:, 1])
    intersection_x2 = np.minimum(preds[:, 2], gts[:, 2])
    intersection_y2 = np.minimum(preds[:, 3], gts[:, 3])

    intersection_w = np.maximum(0,  intersection_x2 - intersection_x1)
    intersection_h = np.maximum(0,  intersection_y2 - intersection_y1)
    intersection_area = intersection_w * intersection_h

    area_preds = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

    union_area = area_preds + area_gts - intersection_area

    iou = intersection_area / (union_area + 1e-7) 

    return iou

# DATASET DETECTION
class BaselineDetectionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.folder = img_dir
        self.transform = transform
        self.images = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert("RGB")
        box = parse_label(img_name)
        scales = torch.tensor([x_scale, y_scale, x_scale, y_scale])
        box = box * scales
        box = box / torch.tensor([W_resize, H_resize, W_resize, H_resize])
        box = box.clamp(0, 1)
        if self.transform:
            image = self.transform(image)
        return image, box

# DATASET RECOGNITION
class BaselineRecognitionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.folder = img_dir
        self.transform = transform
        self.images = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = os.path.join(self.folder, image_name)
        image = Image.open(img_path).convert('RGB')
        x1, y1, x2, y2 = parse_label(image_name)
        image = image.crop((int(x1), int(y1), int(x2), int(y2)))
        if self.transform:
            image = self.transform(image)
        label_str = image_name.split('-')[-3]
        label = torch.tensor(list(map(int, label_str.split('_'))), dtype=torch.long)
        return image, label

# MODELS
class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=None)
        state_dict = torch.load("models/baseline/resnet34.pth", map_location=device)
        resnet.load_state_dict(state_dict)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.regressor(x)

class RecognitionModel(nn.Module):
    def __init__(self, num_classes_list):
        super().__init__()
        base_model = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifiers = nn.ModuleList([nn.Linear(512, n) for n in num_classes_list])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return [clf(x) for clf in self.classifiers]

# TRAINING FUNZIONI
def plot_epoch_losses(losses, title, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    filename = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()

def train_detection(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    model.train()
    scaler = GradScaler()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for imgs, targets in tqdm(train_loader, desc=f"[Detection] Epoch {epoch+1}/{epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        train_avg = epoch_loss / len(train_loader)
        val_avg = validate_detection(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_avg:.6f}, Val Loss={val_avg:.6f}")

        # Saving best and last model
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), os.path.join(save_path, "resnet34_detection_best.pth"))
        torch.save(model.state_dict(), os.path.join(save_path, "resnet34_detection_last.pth"))

        train_losses.append(train_avg)
        val_losses.append(val_avg)
    plot_epoch_losses(train_losses, title="Detection Train Loss", save_dir=save_path)
    plot_epoch_losses(val_losses, title="Detection Val Loss", save_dir=save_path)

def train_recognition(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    model.train()
    scaler = GradScaler()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"[Recognition] Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = sum(criterion(out, labels[:, i]) for i, out in enumerate(outputs))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        train_avg = epoch_loss / len(train_loader)
        val_avg = validate_recognition(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_avg:.6f}, Val Loss={val_avg:.6f}")

        # Saving best and last model
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), os.path.join(save_path, "resnet34_recognition_best.pth"))

        torch.save(model.state_dict(), os.path.join(save_path, "resnet34_recognition_last.pth"))
        train_losses.append(train_avg)
        val_losses.append(val_avg)

    plot_epoch_losses(train_losses, title="Recognition Train Loss", save_dir=save_path)
    plot_epoch_losses(val_losses, title="Recognition Val Loss", save_dir=save_path)

@torch.no_grad()
def validate_detection(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_recognition(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = sum(criterion(out, labels[:, i]) for i, out in enumerate(outputs))
        total_loss += loss.item()
    return total_loss / len(dataloader)    

def complete_pipeline_test(model_detection, model_recognition, dataset_path, iou_threshold=0.6, batch_size=BATCH_SIZE):
    model_detection.eval()
    model_recognition.eval()

    model_detection.to(device)
    model_recognition.to(device)

    dataset = BaselineDetectionDataset(dataset_path, transform=transform_detection)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    total_images = 0
    correct_predictions = 0
    skipped_images = 0
    valid_images = 0

    char_total = 0
    char_correct = 0
    iou_list = []

    with torch.no_grad():
        for batch_idx, (batch_imgs, batch_true_bboxes) in enumerate(tqdm(dataloader, desc="Testing total pipeline")):
            batch_imgs = batch_imgs.to(device)
            batch_preds = model_detection(batch_imgs).cpu().numpy()
            batch_true_bboxes = batch_true_bboxes.numpy()
            batch_size_actual = batch_imgs.size(0)

            # Denormalizza le bbox predette
            batch_preds[:, [0, 2]] *= W_resize
            batch_preds[:, [1, 3]] *= H_resize
            abs_true = batch_true_bboxes * np.array([W_resize, H_resize, W_resize, H_resize])

            # Calcolo IoU batch
            ious = compute_iou(batch_preds, abs_true)
            iou_list.extend(ious)

            for i in range(batch_size_actual):
                total_images += 1

                if ious[i] < iou_threshold:
                    skipped_images += 1
                    continue

                valid_images += 1
                global_idx = batch_idx * batch_size + i
                img_path = os.path.join(dataset_path, dataset.images[global_idx])
                original_img = Image.open(img_path).convert("RGB")

                # Bounding box predetta in coordinate originali
                pred_bbox = batch_preds[i]
                x1, y1, x2, y2 = map(int, pred_bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W_resize, x2), min(H_resize, y2)

                x1 = x1 / x_scale
                y1 = y1 / y_scale
                x2 = x2 / x_scale
                y2 = y2 / y_scale

                cropped_img = original_img.crop((x1, y1, x2, y2))
                cropped_img = transform_recognition(cropped_img).unsqueeze(0).to(device)

                outputs = model_recognition(cropped_img)
                pred = torch.stack([torch.argmax(out, dim=1) for out in outputs], dim=1).squeeze(0)

                filename = dataset.images[global_idx]
                label_str = filename.split('-')[-3]
                true_label = torch.tensor(list(map(int, label_str.split('_')))).to(device)

                # Accuracy intera targa
                if torch.equal(pred, true_label):
                    correct_predictions += 1

                # Accuracy carattere per carattere
                char_total += len(true_label)
                char_correct += (pred == true_label).sum().item()

    accuracy_pipeline = correct_predictions / valid_images if valid_images > 0 else 0
    accuracy_char = char_correct / char_total if char_total > 0 else 0
    mean_iou = np.mean(iou_list) if iou_list else 0

    print(f"Total number of samples processed: {total_images}")
    print(f"Valid bounding boxes (IoU > {iou_threshold}): {valid_images}")
    print(f"Skipped (IoU <= {iou_threshold}): {skipped_images}")
    print(f"Correct full license plates: {correct_predictions}")
    print(f"Character-level accuracy: {accuracy_char * 100:.2f}%")
    print(f"Pipeline accuracy (full plate, IoU > {iou_threshold}): {accuracy_pipeline * 100:.2f}%")
    print(f"Mean IoU over all samples: {mean_iou:.4f}")

# MAIN ENTRY POINT
def main():
    print("[INFO] Creating save directory...")
    os.makedirs(save_path, exist_ok=True)

    # Detection
    print("[INFO] Loading Detection Dataset...")
    train_det_dataset = BaselineDetectionDataset(train_folder, transform=transform_detection)
    val_det_dataset = BaselineDetectionDataset(val_folder, transform=transform_detection)

    print("[INFO] Creating Detection DataLoader...")
    det_loader = DataLoader(train_det_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24, pin_memory=True)
    val_det_loader = DataLoader(val_det_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)


    print("[INFO] Initializing Detection Model...")
    det_model = DetectionModel()
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs for detection")
        det_model = nn.DataParallel(det_model)
        det_model = det_model.to(device)
    else:
        print("[INFO] Using single GPU or CPU for detection")

    det_optim = optim.Adam(det_model.parameters(), lr=1e-3)
    det_crit = nn.SmoothL1Loss()

    print("[INFO] Starting Detection Training...")
    train_detection(det_model, det_loader, val_det_loader, det_optim, det_crit, epochs=N_EPOCHS)

    print("[INFO] Saving Detection Model...")
    torch.save(det_model.state_dict(), os.path.join(save_path, "resnet34_detection.pth"))

    # Recognition
    print("[INFO] Loading Recognition Dataset...")
    train_rec_dataset = BaselineRecognitionDataset(train_folder, transform=transform_recognition)
    val_rec_dataset = BaselineRecognitionDataset(val_folder, transform=transform_recognition)

    print("[INFO] Creating Recognition DataLoader...")
    rec_loader = DataLoader(train_rec_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24, pin_memory=True)
    val_rec_loader = DataLoader(val_rec_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    print("[INFO] Initializing Recognition Model...")
    num_classes = [len(provinces), len(alphabets)] + [len(ads)] * 5
    rec_model = RecognitionModel(num_classes)
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs for recognition")
        rec_model = nn.DataParallel(rec_model)
        rec_model = rec_model.to(device)
    else:
        print("[INFO] Using single GPU or CPU for recognition")

    rec_optim = optim.Adam(rec_model.parameters(), lr=1e-4)
    rec_crit = nn.CrossEntropyLoss()

    print("[INFO] Starting Recognition Training...")
    train_recognition(rec_model, rec_loader, val_rec_loader, rec_optim, rec_crit, epochs=N_EPOCHS)

    print("[INFO] Saving Recognition Model...")
    torch.save(rec_model.state_dict(), os.path.join(save_path, "resnet34_recognition.pth"))

    print(f"[INFO] Training complete. Used GPUs: {torch.cuda.device_count()}")

    # TESTING
    print("[INFO] Loading models for final evaluation on test subsets...")
    det_model.load_state_dict(torch.load(os.path.join(save_path, "resnet34_detection.pth")))
    rec_model.load_state_dict(torch.load(os.path.join(save_path, "resnet34_recognition.pth")))
    det_model.eval()
    rec_model.eval()
    base_test_path = "/leonardo_scratch/large/userexternal/gdaddari/dataset/CCPD_YOLO"
    for subset in test_subsets:
        print(f"\n[INFO] Running full pipeline evaluation on subset: {subset}")
        subset_path = os.path.join(base_test_path, subset, "images", "test")
        complete_pipeline_test(
            model_detection=det_model,
            model_recognition=rec_model,
            dataset_path=subset_path,
            iou_threshold=0.6,
            batch_size=BATCH_SIZE
        )

if __name__ == "__main__":
    main()