# -------- Standard Library --------- #
import os
import io
import random
import shutil
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time

# -------- Other Libraries --------- #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm

# -------- PyTorch --------- #
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, random_split

# -------- Torchvision --------- #
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import transforms, models, datasets


os.environ['WANDB_MODE'] = 'disabled'

# -------- General paths --------- #
DATASET_PATH = "dataset/CCPD2019"
TRAINING_PATH = "dataset/CCPD2019/ccpd_base"
TEST_DIR = "ccpd_challenge"
TEST_PATH = f"dataset/CCPD_YOLO/{TEST_DIR}/images/test" 

# -------- YOLO paths --------- #
DATASET_PATH_YOLO = "dataset/CCPD_YOLO"
TRAINING_CONFIG_YOLO = "dataset/ccpd_2019.yaml"
PROJECT_PATH = '/leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs'
YOLO_MODEL_PATH = 'src/YOLO/runs/train/weights/best.pt'

# -------- Dataset Globals --------- #
IMG_WIDTH = 720
IMG_HEIGHT = 1160
CLASS_ID = 0 

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

charset = PROVINCES + [c for c in ALPHABETS if c not in PROVINCES] + [str(i) for i in range(10)]
charset = list(dict.fromkeys(charset)) 

# -------- BASELINE specific globals --------- #
W_RESIZE = 224
H_RESIZE =224
X_SCALE = W_RESIZE/IMG_WIDTH
Y_SCALE = H_RESIZE/IMG_HEIGHT

transform_detection = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

transform_recognition = transforms.Compose([
    transforms.Resize((64, 128)),  
    transforms.ToTensor(),
])

# -------- GPU support --------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Class definitions --------- #
class CCPDImage:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.valid = self._parse()

    def _parse(self):
        parts = self.filename.stem.split('-')
        if len(parts) != 7:
            return False
        self.parts = parts
        return True

    @property
    def plate_code(self):
        try:
            code = list(map(int, self.parts[4].split('_')))
            return code
        except Exception:
            return None

    @property
    def plate_str(self):
        try:
            code = self.plate_code
            province = PROVINCES[code[0]]
            letter = ALPHABETS[code[1]]
            tail = ''.join(ADS[i] for i in code[2:])
            return province + letter + tail
        except Exception:
            return "INVALID"

    @property
    def bbox_absolute(self):
        try:
            bbox_str = self.parts[2]
            x1y1_str, x2y2_str = bbox_str.split('_')
            x1, y1 = map(int, x1y1_str.split('&'))
            x2, y2 = map(int, x2y2_str.split('&'))
            return torch.tensor([x1, y1, x2, y2], dtype=torch.float)
        except Exception:
            return None

    @property
    def bbox_yolo(self):
        try:
            x1, y1, x2, y2 = self.bbox_absolute
            bbox_width = abs(x2 - x1)
            bbox_height = abs(y2 - y1)
            x_center = x1 + bbox_width / 2.0
            y_center = y1 + bbox_height / 2.0

            x_center /= IMG_WIDTH
            y_center /= IMG_HEIGHT
            bbox_width /= IMG_WIDTH
            bbox_height /= IMG_HEIGHT

            return (x_center, y_center, bbox_width, bbox_height)
        except Exception:
            return None

    def __repr__(self):
        return f"CCPDImageInfo(plate='{self.plate_str}', valid={self.valid})"
    
class CCPDDataset(Dataset):
    def __init__(self, img_dir, transform=None, task="detection"):
        """
        task: 'detection' | 'recognition'
        """
        self.img_dir = Path(img_dir)
        self.task = task

        # If not transform and we're in recognition task, use FullRobustAugmentation
        if transform is None and task == "recognition":
            self.transform = FullRobustAugmentation()
        else:
            self.transform = transform

        self.image_paths = [p for p in self.img_dir.glob("*.jpg")]
        self.image_objs = [CCPDImage(p) for p in self.image_paths if CCPDImage(p).valid]

    def __len__(self):
        return len(self.image_objs)

    def __getitem__(self, idx):
        img_obj = self.image_objs[idx]
        img_path = img_obj.filename
        image = Image.open(img_path).convert("RGB")

        if self.task == "detection":
            # Image full + bbox
            if self.transform:
                image = self.transform(image)
            bbox = img_obj.bbox_absolute  # torch.tensor([x1, y1, x2, y2])
            return image, bbox

        elif self.task == "recognition":
            x1, y1, x2, y2 = img_obj.bbox_absolute.tolist()
            image = image.crop((x1, y1, x2, y2))

            if self.transform:
                image = self.transform(image)

            plate_code = img_obj.plate_code
            return image, torch.tensor(plate_code)

        else:
            raise ValueError(f"Task '{self.task}' not allowed.")
        
class FullRobustAugmentation:
    def __init__(self):
        self.base = transforms.Compose([
            transforms.Resize((48, 144)),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.3, hue=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])


       
    def __call__(self, img):
        img = self.base(img)  # geometrie e jitter

        if random.random() < 0.5:
            img = self.random_motion_blur(img)

        if random.random() < 0.5:
            factor = random.uniform(0.3, 1.8)
            img = TF.adjust_brightness(img, factor)

        if random.random() < 0.5:
            img = self.random_occlusion(img)

        if random.random() < 0.5:
            img = self.random_compression(img)

        if random.random() < 0.5:
            img = self.add_fog(img)

        return TF.to_tensor(img)


    def random_motion_blur(self, img):
        kernel_size = random.choice([5, 9, 15])
        return img.filter(ImageFilter.GaussianBlur(radius=kernel_size / 5))

    def add_fog(self, img):
        fog = Image.new("RGB", img.size, color=(200, 200, 200))
        return Image.blend(img, fog, alpha=random.uniform(0.1, 0.4))


    def random_occlusion(self, img):
        draw = img.copy()
        w, h = draw.size
        x0 = random.randint(0, w // 2)
        y0 = random.randint(0, h // 2)
        x1 = x0 + random.randint(10, 40)
        y1 = y0 + random.randint(10, 20)
        color = random.choice([(0, 0, 0), (255, 255, 255)])
        for x in range(x0, min(x1, w)):
            for y in range(y0, min(y1, h)):
                draw.putpixel((x, y), color)
        return draw

    def random_compression(self, img):
        buffer = io.BytesIO()
        quality = random.randint(10, 40)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class BaselineModel(nn.Module):
    def __init__(self, num_classes_list, pretrained=True):
        super().__init__()

        # Shared backbone
        resnet = models.resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Fino a conv5_x
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Detection head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

        # Recognition head (multi-head classifier)
        self.classifiers = nn.ModuleList([
            nn.Linear(512, n_classes) for n_classes in num_classes_list
        ])

    def forward_backbone(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # [B, 512]

    def forward_detection(self, x):
        x = self.forward_backbone(x)
        return self.regressor(x)

    def forward_recognition(self, x):
        x = self.forward_backbone(x)
        return [clf(x) for clf in self.classifiers]

    def forward(self, x, mode=None):
        """
        mode: 'detection' or 'recognition'
        """
        if mode == 'detection':
            return self.forward_detection(x)
        elif mode == 'recognition':
            return self.forward_recognition(x)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
  
class Metrics:
    def __init__(self, task):
        self.task = task
        self.reset()

    def reset(self):
        self.total_iou = []
        self.correct_chars = 0
        self.total_chars = 0
        self.correct_chars_wo_first = 0
        self.total_time = 0
        self.total_samples = 0
    
    @staticmethod
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

    def update_detection(self, preds_abs, targets):
        ious = self.compute_iou(preds_abs, targets)  #IoU for each pair
        for iou in ious:
            self.total_iou.append(iou)          

    def update_recognition(self, outputs, targets):
        batch_size = targets.size(0)
        for i, output in enumerate(outputs):
            pred_label = output.argmax(dim=1)
            correct = (pred_label == targets[:, i]).sum().item()
            self.correct_chars += correct

            if i > 0:  # Without chinese char
                self.correct_chars_wo_first += correct

        self.total_chars += batch_size * len(outputs)

    def update_time(self, start_time, end_time, batch_size):
        self.total_time += end_time - start_time
        self.total_samples += batch_size

    def compute(self):
        results = {}
        if self.task == "detection":
            results['IoU'] = np.mean(self.total_iou)
        elif self.task == "recognition":
            results['Accuracy'] = self.correct_chars / self.total_chars
            if self.total_chars > 0:
                results['Accuracy_wo_first'] = self.correct_chars_wo_first / (self.total_chars - self.total_samples)
        results['FPS'] = self.total_samples / self.total_time if self.total_time > 0 else 0
        return results

class Trainer:
    def __init__(self, model, task, device, lr=1e-3, num_classes_list=None):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []

        # Recognizing model
        self.model_type = model.model_type if hasattr(model, "model_type") else "baseline"
        self.model_path = model.model_path if hasattr(model, "model_path") else "" 

        self.set_task(task, num_classes_list)

    def plot_epoch_losses(self, losses, title, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses)+1), losses, label='Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Loss graph saved in: '{save_path}'")

    def set_task(self, task, num_classes_list=None):
        self.task = task
        if task == "detection":
            self.criterion = nn.MSELoss()
            self.criterions = None  # Reset
        elif task == "recognition":
            if num_classes_list is None:
                raise ValueError("num_classes_list needed.")
            self.criterions = [nn.CrossEntropyLoss() for _ in num_classes_list]
            self.criterion = None  # Reset
        else:
            raise ValueError(f"Task not allowed: {task}")
    
    def train(self, dataloader=None, epochs=10, batch_size=50, optimizer="Adam",lr0=1e-3,lrf=1e-5, cos_lr=True,project="runs/train", name="lp_detection", cache="ram"):
        if self.model_type == "yolov5":
            return self._train_yolov5(epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr0=lr0, lrf=lrf, cos_lr=cos_lr, project=project, name=name, cache=cache)
        elif self.model_type == "pdlpr":
            return self._train_pdlpr(dataloader, epochs)
        else:
            return self._train_baseline(dataloader, epochs)

    def _train_baseline(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in tqdm(dataloader, desc=f"[{self.task}] Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                if self.task == "detection":
                    bboxes = targets.to(self.device)
                    norm = torch.tensor([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT]).to(self.device)
                    bboxes = bboxes / norm
                    preds = self.model(images, 'detection')
                    loss = self.criterion(preds, bboxes)
                elif self.task == "recognition":
                    labels = targets.to(self.device)
                    outputs = self.model(images, 'recognition')
                    loss = 0
                    for i, crit in enumerate(self.criterions):
                        loss += crit(outputs[i], labels[:, i])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
            self.losses.append(avg_loss)
        self.plot_epoch_losses(self.losses, f"{self.task.capitalize()} Loss", "results")
        return self.model

    def _train_yolov5(self, epochs=25, batch_size=50, optimizer="Adam", lr0=1e-3, lrf=1e-5, cos_lr=True, project="runs/train", name="lp_detection", cache="ram"):
        YOLOv5_training(
            weights=self.model_path,
            data=DATASET_PATH_YOLO,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=640,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            cos_lr=cos_lr,
            project=project,
            name=name,
            cache=cache
        )
        return self.model

    def _train_pdlpr(self, dataloader, epochs):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

        train_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [PDLPR Train]", unit="batch")
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    output = self.model(images)
                    output = output.permute(0, 2, 1)  # [B, SeqLen, C] -> [B, C, SeqLen]
                    loss = loss_fn(output, targets)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"batch_loss": loss.item()})

            avg_loss = running_loss / len(dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

            torch.save(self.model.state_dict(), f"models/PDLPR/weights/pdlpr_epoch{epoch+1}.pth")

        # Save final model
        torch.save(self.model.state_dict(), "models/PDLPR/weights/pdlpr_final.pth")

        # Plotting
        self.plot_epoch_losses(train_losses, "PDLPR Loss", "models/PDLPR/logs")
        return self.model
    
class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model_type = model.model_type if hasattr(model, "model_type") else "baseline"

    @torch.no_grad()
    def evaluate(self, dataloader, task):
        self.model.eval()
        metrics = Metrics(task=task)

        for images, targets in tqdm(dataloader, desc=f"[Evaluating {task}]"):
            start_time = time.time()
            images = images.to(self.device)

            if task == "detection":
                targets = targets.to(self.device)
                preds = self.model(images, mode='detection')
                preds_abs = preds * torch.tensor([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT], device=self.device)
                metrics.update_detection(preds_abs.cpu().numpy(), targets.cpu().numpy())

            elif task == "recognition":
                targets = targets.to(self.device)
                outputs = self.model(images, mode='recognition')
                metrics.update_recognition(outputs, targets)

            end_time = time.time()
            metrics.update_time(start_time, end_time, images.size(0))

        return metrics.compute()


if __name__ == "__main__":
    # Prepara dataset
    train_dataset_det = CCPDDataset(TRAINING_PATH, transform=transform_detection, task='detection')
    train_loader_det = DataLoader(train_dataset_det, batch_size=32, shuffle=True)

    train_dataset_rec = CCPDDataset(TRAINING_PATH, transform=transform_recognition, task='recognition')
    train_loader_rec = DataLoader(train_dataset_rec, batch_size=32, shuffle=True)

    test_dataset_det = CCPDDataset(TEST_PATH, transform=transform_detection, task='detection')
    test_loader_det = DataLoader(test_dataset_det, batch_size=32, shuffle=True)

    test_dataset_rec = CCPDDataset(TEST_PATH, transform=transform_recognition, task='recognition')
    test_loader_rec = DataLoader(test_dataset_rec, batch_size=32, shuffle=True)

    # Costanti
    num_classes_list = [len(PROVINCES), len(ALPHABETS)] + [len(ADS)] * 5 

    # -------------------------
    # Training detection model
    # -------------------------
    det_model = BaselineModel(num_classes_list=num_classes_list)
    det_model.load_state_dict(torch.load("models/baseline/resnet34_detection_best.pth"))
    det_trainer = Trainer(det_model, task="detection", device=DEVICE)
    det_model = det_trainer.train(train_loader_det, epochs=20)


    # -------------------------
    # Training recognition model
    # -------------------------
    rec_model = BaselineModel(num_classes_list=num_classes_list)
    rec_model.load_state_dict(torch.load("models/baseline/resnet34_recognition_best.pth"))
    rec_trainer = Trainer(rec_model, task="recognition", device=DEVICE)
    rec_model = rec_trainer.train(train_loader_rec, epochs=20)


    # -------------------------
    # Evaluation detection
    # -------------------------
    evaluator = Evaluator(det_model, device=DEVICE)
    metrics_det = evaluator.evaluate(test_loader_det, task="detection")
    print("Detection Results:", metrics_det)

    # -------------------------
    # Evaluation recognition
    # -------------------------
    evaluator = Evaluator(rec_model, device=DEVICE)
    metrics_rec = evaluator.evaluate(test_loader_rec, task="recognition")
    print("Recognition Results:", metrics_rec)