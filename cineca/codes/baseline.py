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
from collections import OrderedDict

# -------- PyTorch --------- #
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, is_initialized, barrier
from torch.utils.data.distributed import DistributedSampler

# -------- Torchvision --------- #
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import models, datasets


os.environ['WANDB_MODE'] = 'disabled'

# -------- General paths --------- #
DATASET_PATH = "dataset/CCPD2019"
#TRAINING_PATH = "dataset/CCPD2019/ccpd_base"
TRAINING_PATH = os.path.join(os.environ["SCRATCH"], "dataset/CCPD2019/ccpd_base")
TEST_DIR = "ccpd_blur"
#TEST_PATH = f"$SCRATCH/dataset/CCPD_YOLO/{TEST_DIR}/images/test"
TEST_PATH = os.path.join(os.environ["SCRATCH"], "dataset", "CCPD_YOLO", TEST_DIR, "images", "test")

# -------- YOLO Globals --------- #
DATASET_PATH_YOLO = "dataset/CCPD_YOLO"
TRAINING_CONFIG_YOLO = "dataset/ccpd_2019.yaml"
PROJECT_PATH = '/leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs'
YOLO_MODEL_PATH = 'src/YOLO/runs/train/weights/best.pt'
YOLO_IMG_SIZE = 640
transform_yolo = T.Compose([
    T.Resize((YOLO_IMG_SIZE, YOLO_IMG_SIZE))
])

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

transform_recognition = T.Compose([
    T.Resize((64, 128)),  
    T.ToTensor(),
])


# -------- PDLPR specific globals --------- #
class SimplePlateTokenizer:
    def __init__(self, charset):
        self.char2idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 = PAD
        self.char2idx['<PAD>'] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}
    def encode(self, text):
        for c in text:
            if c not in self.char2idx:
                print(f"[Tokenizer Warning] Carattere '{c}' non nel charset! Verrà codificato come PAD (0)")
        return [self.char2idx.get(c, 0) for c in text]
    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '') for i in indices if i != 0])
    def vocab_size(self):
        return len(self.char2idx)

tokenizer = SimplePlateTokenizer(charset)
num_classes = tokenizer.vocab_size()
seq_len = 8  # maximum car plate length

# -------- GPU support --------- #
def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank

def cleanup_ddp():
    dist.destroy_process_group()

rank = setup_ddp()
DEVICE = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

def disp(msg):
    if rank == 0:
        print(msg)
    

# -------- Class definitions --------- #
class YOLOv5():
    pass
class PDLPR():
    pass
    
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
    def bbox_normalized_baseline(self): #Normalized for the baseline model
        try:
            bbox_str = self.parts[2]
            x1y1_str, x2y2_str = bbox_str.split('_')
            x1, y1 = map(int, x1y1_str.split('&'))
            x2, y2 = map(int, x2y2_str.split('&'))
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            box /= np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
            return torch.tensor(box, dtype=torch.float32)
        except Exception:
            return None
        
    @property
    def bbox_absolute(self): #NOT Normalized for the baseline model
        try:
            bbox_str = self.parts[2]
            x1y1_str, x2y2_str = bbox_str.split('_')
            x1, y1 = map(int, x1y1_str.split('&'))
            x2, y2 = map(int, x2y2_str.split('&'))
            return x1, y1, x2, y2
        except Exception:
            return None

    @property
    def bbox_yolo(self): #Normalized in yolo format
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
    def __init__(self, img_dir, transform=None, task="detection", model="baseline"):
        """
        task: 'detection' | 'recognition' | 'end_to_end'
        """
        self.img_dir = Path(img_dir)
        self.task = task
        self.model = model

        # If not transform and we're in recognition task, use FullRobustAugmentation
        if transform is None and task == "recognition":
            self.transform = FullRobustAugmentation()
        else:
            if transform == "test":
                self.transform = T.ToTensor()
            else:
                self.transform = transform

        self.image_paths = [p for p in self.img_dir.glob("*.jpg")]
        self.image_objs = [CCPDImage(p) for p in self.image_paths if CCPDImage(p).valid]

    def __len__(self):
        return len(self.image_objs)
    
    def set_task(self, task, model):
        assert task in {"detection", "recognition", "end_to_end"}, "Task not allowed"
        assert model in {"baseline","yolov5","pdlpr"}, "Model not supported"
        self.model = model
        self.task = task

    def __getitem__(self, idx):
        img_obj = self.image_objs[idx]
        img_path = img_obj.filename
        image = Image.open(img_path).convert("RGB")

        if self.task == "detection":
            # Image full + bbox
            if self.transform:
                image = self.transform(image)
            if self.model == "baseline":
                bbox = img_obj.bbox_normalized_baseline
            elif self.model == "yolov5":
                bbox = img_obj.bbox_yolo
            return image, bbox

        elif self.task == "recognition":
            # Valid both for baseline and PDLPR
            x1, y1, x2, y2 = img_obj.bbox_absolute
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            image = image.crop((x1, y1, x2, y2))

            if self.transform:
                image = self.transform(image)

            plate_code = img_obj.plate_code
            return image, torch.tensor(plate_code)

    
        elif self.task == "end_to_end":
            if self.transform:
                image = self.transform(image)
            if self.model == "baseline":
                bbox = img_obj.bbox_normalized_baseline
                plate_code = torch.tensor(img_obj.plate_code)
            elif self.model == "yolov5":
                bbox = img_obj.bbox_yolo
                plate_str = img_obj.plate_str
                encoded_plate = tokenizer.encode(plate_str)
                padded_encoded_plate = encoded_plate + [0] * (seq_len - len(encoded_plate))
                plate_code = torch.tensor(padded_encoded_plate[:seq_len])
            
            return image, bbox, plate_code

        else:
            raise ValueError(f"Task '{self.task}' not allowed.")
        
class FullRobustAugmentation:
    def __init__(self):
        self.base = T.Compose([
            T.Resize((48, 144)),
            T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.3, hue=0.1),
            T.RandomRotation(degrees=30),
            T.RandomAffine(degrees=0, shear=10),
            T.RandomPerspective(distortion_scale=0.4, p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
    def __init__(self, num_classes_list, mode="detection", loading_from_path=None, resnet_weights_path=None):
        super().__init__()

        # Detection backbone
        if mode == "detection":
            if resnet_weights_path:
                resnet = models.resnet34(weights=None)     
                state_dict = torch.load(resnet_weights_path, map_location="cpu")
                resnet.load_state_dict(state_dict)
            else:
                if loading_from_path is not None:
                    resnet = models.resnet34(weights=None)
                else:
                    resnet = models.resnet34(weights="DEFAULT")
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 4),
                nn.Sigmoid()  
            )
            # Recognition backbone
        elif mode == "recognition":
            if resnet_weights_path:
                resnet = models.resnet34(weights=None)
                state_dict = torch.load(resnet_weights_path, map_location="cpu")
                resnet.load_state_dict(state_dict)
            else:
                resnet = models.resnet34(weights="DEFAULT")
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  
            self.classifiers = nn.ModuleList([
                nn.Linear(512, n_classes) for n_classes in num_classes_list
            ])
        else:
            raise ValueError("mode not supported. Use either detection or recognition.")
        
        if loading_from_path is not None:
            state_dict = torch.load(loading_from_path, map_location="cpu")
            # Fix DataParallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            self.load_state_dict(new_state_dict)

    def forward_detection(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x

    def forward_recognition(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        outputs = [clf(x) for clf in self.classifiers]
        return outputs

    def forward(self, x, mode=None):
        """
        mode: 'detection' or 'recognition'
        """
        if mode == 'detection':
            return self.forward_detection(x)
        elif mode == 'recognition':
            return self.forward_recognition(x)
        else:
            raise ValueError("mode not supported. Use either detection or recognition.")

class Metrics:
    def __init__(self, task):
        self.task = task
        self.reset()

    def reset(self):
        self.correct_detections_iou7 = 0
        self.correct_recognitions_iou6 = 0 
        self.correct_plates_full = 0 
        self.mean_iou = []
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

    def update_detection(self, preds_abs, targets, ious=None):
        self.total_samples += targets.shape[0]
        if ious is None:
            ious = self.compute_iou(preds_abs, targets)
        self.current_batch_ious = ious
        self.mean_iou.extend(ious.tolist())
        self.correct_detections_iou7 += (np.array(ious) > 0.7).sum()      

    def update_recognition(self, outputs, targets):
        if outputs is None or len(outputs) == 0:
            self.total_samples += targets.shape[0]
            return

        # predictions: [B] -> torch.stack -> [B, T]
        pred_labels = torch.stack([logits.argmax(dim=1) for logits in outputs], dim=1)
        targets_valid = targets.to(pred_labels.device)
        num_valid_predictions = pred_labels.size(0)
        assert num_valid_predictions == targets_valid.size(0) #Check

        for i in range(num_valid_predictions):
            is_full = torch.equal(pred_labels[i], targets_valid[i])
            is_wo_first = torch.equal(pred_labels[i][1:], targets_valid[i][1:])

            if is_full:
                self.correct_recognitions_iou6 += 1
                self.correct_plates_full += 1
            if is_wo_first:
                self.correct_chars_wo_first += 1


    def update_time(self, start_time, end_time):
        self.total_time += end_time - start_time
    
    def compute(self):
        if self.total_samples == 0:
            return {}
            
        results = {
            'FPS': float(self.total_samples / self.total_time) if self.total_time > 0 else 0.0,
            'Mean_IoU': float(np.mean(self.mean_iou)) if self.mean_iou else 0.0,
            'Detection_Accuracy_IoU_0.7': float(100 * self.correct_detections_iou7 / self.total_samples),
            'Recognition_Accuracy_IoU_0.6': float(100 * self.correct_recognitions_iou6 / self.total_samples),
            'Plate_Accuracy_Full': float(100 * self.correct_plates_full / self.total_samples),
            'Plate_Accuracy(-Fisrt_Char)': float(100 * self.correct_chars_wo_first / self.total_samples),
        }

        return {
            k: round(v, 4) if k == 'Mean_IoU' else round(v, 2)
            for k, v in results.items()
        }

class Trainer:
    def __init__(self, model, task, device, lr=1e-3, num_classes_list=None):
        self.model = model.to(device)
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[device])
        self.task = task
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []

        # Recognizing model
        self.model_type = model.model_type if hasattr(model, "model_type") else "baseline"
        self.model_path = model.model_path if hasattr(model, "model_path") else "" 

        self.set_task(task, num_classes_list)
        
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

    def plot_epoch_losses(self, train_losses, val_losses, title, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
        if val_losses:
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(save_path)
        plt.close()
        disp(f"Loss graph saved in: '{save_path}'")
    
    def train(self, dataloader=None, epochs=10, batch_size=50, optimizer="Adam",lr0=1e-3,lrf=1e-5, cos_lr=True,project="runs/train", name="lp_detection", cache="ram"):
        if self.model_type == "yolov5":
            return self._train_yolov5(epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr0=lr0, lrf=lrf, cos_lr=cos_lr, project=project, name=name, cache=cache)
        elif self.model_type == "pdlpr":
            return self._train_pdlpr(dataloader, epochs)
        else:
            return self._train_baseline(dataloader, epochs)

    def _train_baseline(self, dataloader, epochs, val_dataloader=None):
        self.model.train()
        best_loss = float('inf')
        best_model_path = f"models/baseline/best_{self.task}_model_3.pth"
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            total_loss = 0
            for images, targets in tqdm(dataloader, desc=f"[{self.task}] Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                if self.task == "detection":
                    bboxes = targets.to(self.device)
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
            avg_train_loss = total_loss / len(dataloader)
            train_losses.append(avg_train_loss)
            disp(f"Epoch {epoch+1} - Loss: {avg_train_loss:.8f}")
            
            #Validation
            avg_val_loss = None
            if val_dataloader:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for images, targets in val_dataloader:
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
                            loss = sum(crit(outputs[i], labels[:, i]) for i, crit in enumerate(self.criterions))

                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)
                disp(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
                self.model.train()

        # Saving the best model
        current_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(self.model.state_dict(), best_model_path)
            disp(f"Best model saved. Loss: {best_loss:.4f}")

        self.plot_epoch_losses(train_losses, val_losses, f"{self.task.capitalize()} Loss", "models/baseline/train")
        return self.model

    def _train_yolov5(self, epochs=25, batch_size=50, optimizer="Adam", lr0=1e-3, lrf=1e-5, cos_lr=True, project="runs/train", name="lp_detection", cache="ram"):
        pass

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
            disp(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

            torch.save(self.model.state_dict(), f"models/PDLPR/weights/pdlpr_epoch{epoch+1}.pth")

        # Save final model
        torch.save(self.model.state_dict(), "models/PDLPR/weights/pdlpr_final.pth")

        # Plotting
        self.plot_epoch_losses(train_losses, "PDLPR Loss", "models/PDLPR/logs")
        return self.model


class Evaluator:
    def __init__(self, det_model,rec_model, device):
        self.det_model = det_model.to(device)
        self.rec_model = rec_model.to(device)
        self.device = device
        
        self.transform_recognition = transform_recognition
        self.transform_detection = transform_detection

        if isinstance(self.det_model, torch.nn.parallel.DistributedDataParallel):
            self.det_model = self.det_model.module
        if isinstance(self.rec_model, torch.nn.parallel.DistributedDataParallel):
            self.rec_model = self.rec_model.module
        
    @torch.no_grad()
    def evaluate(self, dataloader, task):
        self.det_model.eval()
        self.rec_model.eval()

        metrics = Metrics(task="end_to_end")
        is_main_process = not is_initialized() or get_rank() == 0
        iterator = tqdm(enumerate(dataloader), desc=f"[Evaluating end-to-end]", total=len(dataloader)) if is_main_process else enumerate(dataloader)

        for batch_idx, (images, gt_bboxes, plate_labels) in iterator:
            start_time = time.time()
            images = images.to(self.device)
            
            # Detection
            if isinstance(self.det_model, YOLOv5):
                gt_bboxes_list = gt_bboxes
                gt_bboxes = torch.stack(gt_bboxes_list, dim=1).to(self.device)
                pred_bboxes_norm = self.det_model(images)
                
                # Conversion from YOLO to x1y1x2y2
                xc, yc, w, h = gt_bboxes.T
                gt_x1, gt_y1, gt_x2, gt_y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                gt_bboxes_norm_x1y1 = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)
                
                # Computing abs coordinates 640x640
                scale_tensor = torch.tensor([YOLO_IMG_SIZE] * 4, device=self.device, dtype=torch.float32)
                pred_bboxes_abs = pred_bboxes_norm * scale_tensor
                gt_bboxes_abs = gt_bboxes_norm_x1y1 * scale_tensor
                
                # Memorize dim for rescaling
                resize_w, resize_h = YOLO_IMG_SIZE, YOLO_IMG_SIZE

            else: # Baseline
                gt_bboxes = gt_bboxes.to(self.device)
                pred_bboxes_norm = self.det_model(images, mode='detection')
                
                # Computing abs coordinates 224x224
                scale_tensor = torch.tensor([W_RESIZE, H_RESIZE, W_RESIZE, H_RESIZE], device=self.device)
                pred_bboxes_abs = pred_bboxes_norm * scale_tensor
                gt_bboxes_abs = gt_bboxes * scale_tensor
                
                # Memorize dim for rescaling
                resize_w, resize_h = W_RESIZE, H_RESIZE
            
            # IoU
            ious = Metrics.compute_iou(pred_bboxes_abs.cpu().numpy(), gt_bboxes_abs.cpu().numpy())
            metrics.update_detection(pred_bboxes_abs.cpu().numpy(), gt_bboxes_abs.cpu().numpy(), ious)

            # Recognition
            valid_mask = torch.from_numpy(ious) > 0.6
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                bboxes_for_cropping_resized = pred_bboxes_abs[valid_mask]
                
                cropped_images_for_rec = []
                labels_for_rec = []

                for i, bbox_resized in zip(valid_indices, bboxes_for_cropping_resized):
                    # Original image path
                    original_dataset_idx = batch_idx * dataloader.batch_size + i.item()
                    if original_dataset_idx >= len(dataloader.dataset): continue
                    img_obj = dataloader.dataset.image_objs[original_dataset_idx]
                    original_pil_img = Image.open(img_obj.filename).convert("RGB")
                    
                    # Rescale bbox
                    x1_res, y1_res, x2_res, y2_res = bbox_resized
                    scale_x = IMG_WIDTH / resize_w
                    scale_y = IMG_HEIGHT / resize_h
                    x1_orig = int(x1_res * scale_x)
                    y1_orig = int(y1_res * scale_y)
                    x2_orig = int(x2_res * scale_x)
                    y2_orig = int(y2_res * scale_y)

                    # Cropping original img
                    x1_orig, y1_orig = max(0, x1_orig), max(0, y1_orig)
                    x2_orig, y2_orig = min(IMG_WIDTH, x2_orig), min(IMG_HEIGHT, y2_orig)
                    if x1_orig >= x2_orig or y1_orig >= y2_orig: continue #check
                    cropped_pil = original_pil_img.crop((x1_orig, y1_orig, x2_orig, y2_orig))
                    transformed_crop = self.transform_recognition(cropped_pil)
                    cropped_images_for_rec.append(transformed_crop)
                    labels_for_rec.append(plate_labels[i])

                if cropped_images_for_rec:
                    cropped_batch = torch.stack(cropped_images_for_rec).to(self.device)
                    if isinstance(self.rec_model, PDLPR):
                        outputs = self.rec_model(cropped_batch)
                    else:
                        outputs = self.rec_model(cropped_batch, mode='recognition')

                    labels_tensor = torch.stack(labels_for_rec).to(cropped_batch.device)
                    metrics.update_recognition(outputs, labels_tensor)
                
            end_time = time.time()
            metrics.update_time(start_time, end_time)

        if is_initialized():
            barrier()

        if is_main_process:
            return metrics.compute()
        else:
            return None



def main():

    # Datasets
    #train_dataset_det = CCPDDataset(TRAINING_PATH, transform=transform_detection, task='detection', model="baseline")
    #train_sampler_det = DistributedSampler(train_dataset_det)
    #train_loader_det = DataLoader(train_dataset_det, batch_size=256, sampler=train_sampler_det, num_workers=8)
    #disp(f"Train detection: {len(train_dataset_det)} images")

    #train_dataset_rec = CCPDDataset(TRAINING_PATH, transform=transform_recognition, task='recognition', model="baseline")
    #train_sampler_rec = DistributedSampler(train_dataset_rec)
    #train_loader_rec = DataLoader(train_dataset_rec, batch_size=256, sampler=train_sampler_rec, num_workers=8)
    #disp(f"Train recognition: {len(train_dataset_rec)} images")

    test_dataset = CCPDDataset(TEST_PATH, transform=transform_detection, task='end_to_end', model="baseline")
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
    disp(f"Test dataset (end-to-end): {len(test_dataset)} images")


    # Cnst
    num_classes_list = [len(PROVINCES), len(ALPHABETS)] + [len(ADS)] * 5 

    # -------------------------
    # Training detection model
    # -------------------------
    det_model = BaselineModel(num_classes_list=num_classes_list, mode="detection", loading_from_path="models/baseline/best_detection_model.pth")
    #det_model = BaselineModel(num_classes_list=num_classes_list, resnet_weights_path="models/baseline/resnet34.pth", mode="detection")
    #det_trainer = Trainer(det_model, task="detection", device=DEVICE)
    #det_model = det_trainer.train(train_loader_det, epochs=17)


    # -------------------------
    # Training recognition model
    # -------------------------
    rec_model = BaselineModel(num_classes_list=num_classes_list, loading_from_path="models/baseline/best_recognition_model.pth", mode="recognition")
    #rec_model = BaselineModel(num_classes_list=num_classes_list, resnet_weights_path="models/baseline/resnet34.pth", mode="recognition")
    #rec_trainer = Trainer(rec_model, num_classes_list=num_classes_list, task="recognition", device=DEVICE)
    #rec_model = rec_trainer.train(train_loader_rec, epochs=17)


    # -------------------------
    # Evaluation detection and recognition (end-to-end)
    # -------------------------
    evaluator = Evaluator(det_model=det_model, rec_model=rec_model, device=DEVICE)
    metrics_rec = evaluator.evaluate(test_loader, task="end_to_end")
    disp(f"End-to-End Results:{metrics_rec}")

    cleanup_ddp()
if __name__ == "__main__":
    main()
    