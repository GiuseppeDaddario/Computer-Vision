# -------- Standard Library --------- #
import os
import sys
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
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from collections import OrderedDict
import copy

# -------- PyTorch --------- #
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, is_initialized, barrier
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

# -------- Torchvision --------- #
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import models, datasets
from torchvision.utils import save_image

# -------- Local Imports --------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import YOLOv5_training, non_max_suppression, attempt_load

os.environ['WANDB_MODE'] = 'disabled'

# -------- General paths --------- #
DATASET_PATH = "dataset/CCPD2019"
#TRAINING_PATH = "dataset/CCPD2019/ccpd_base"
TRAINING_PATH = os.path.join(os.environ["SCRATCH"], "dataset/CCPD_YOLO/ccpd_base/images/train")
VALIDATION_PATH = os.path.join(os.environ["SCRATCH"], "dataset/CCPD_YOLO/ccpd_base/images/val")
TEST_DIR = "ccpd_challenge"
#TEST_PATH = f"$SCRATCH/dataset/CCPD_YOLO/{TEST_DIR}/images/test"


# -------- YOLO Globals --------- #
DATASET_PATH_YOLO = "dataset/CCPD_YOLO"
TRAINING_CONFIG_YOLO = "dataset/ccpd_2019.yaml"
PROJECT_PATH = '/leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs'
YOLO_MODEL_PATH = 'src/YOLO/runs/train/weights/best.pt'

YOLO_IMG_SIZE = 640
transform_yolo = T.Compose([
    T.Resize((YOLO_IMG_SIZE, YOLO_IMG_SIZE)),
    T.ToTensor()
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
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

rank, world_size = setup_ddp()
DEVICE = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

def disp(msg):
    if rank == 0:
        print(msg)
    

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
    def __init__(self, img_dir, transform=None, task="detection", model="baseline", max_samples=None, for_attack_generation=False):
        """
        task: 'detection' | 'recognition' | 'end_to_end'
        """
        self.img_dir = Path(img_dir)
        self.task = task
        self.model = model
        self.for_attack_generation = for_attack_generation

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
        if max_samples is not None:
            self.image_objs = self.image_objs[:max_samples]

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

        plate_str = img_obj.plate_str
        encoded_plate = tokenizer.encode(plate_str)
        padded_encoded_plate = encoded_plate + [0] * (seq_len - len(encoded_plate))
        plate_target = torch.tensor(padded_encoded_plate[:seq_len], dtype=torch.long)

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
                image = self.transform(image) #img_obj.plate_str

            #plate_code = img_obj.plate_code
            return image, plate_target

    
        elif self.task == "end_to_end":
            if self.transform:
                image = self.transform(image)

            if self.model == "baseline":
                bbox = img_obj.bbox_normalized_baseline
                #plate_code = torch.tensor(img_obj.plate_code)
            elif self.model == "yolov5":
                bbox = img_obj.bbox_yolo
                #plate_str = img_obj.plate_str
                #encoded_plate = tokenizer.encode(plate_str)
                #padded_encoded_plate = encoded_plate + [0] * (seq_len - len(encoded_plate))
                #plate_code = torch.tensor(padded_encoded_plate[:seq_len])
            
            if self.for_attack_generation:
                return image, plate_target, img_obj.filename.name, img_obj.bbox_absolute
            else:
                return image, bbox, plate_target


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


       
    def __call__(self, img, plate_str=None):
        img = self.base(img)  # geometrie e jitter

        heavy_augmentation = False
        if plate_str is not None:
            province_char = plate_str[0]
            if province_char != PROVINCES[0]:  # If it's not the first one
                heavy_augmentation = True

        # Imposta probabilità e fattori in base al flag
        p_motion_blur = 0.8 if heavy_augmentation else 0.5
        p_brightness = 0.8 if heavy_augmentation else 0.5
        p_occlusion = 0.8 if heavy_augmentation else 0.5
        p_compression = 0.8 if heavy_augmentation else 0.5
        p_fog = 0.8 if heavy_augmentation else 0.5 

        if random.random() < p_motion_blur:
            img = self.random_motion_blur(img)

        if random.random() < p_brightness:
            factor = random.uniform(0.3, 1.8)
            img = TF.adjust_brightness(img, factor)

        if random.random() < p_occlusion:
            img = self.random_occlusion(img)

        if random.random() < p_compression:
            img = self.random_compression(img)

        if random.random() < p_fog:
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

class Metrics:
    def __init__(self, task, recognition=False):
        self.task = task
        self.reset(recognition=recognition)

    def reset(self, recognition):
        self.correct_detections_iou7 = 0
        self.correct_plates_full = 0 
        self.mean_iou = []
        self.correct_chars_wo_first = 0
        self.total_time_detection = 0
        self.total_time_pipeline = 0
        self.total_samples = 0
        self.total_recognition_samples = 0
        self.recognition=recognition
    
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
        # Checking if null, both for baseline and PDLPR.
        if (isinstance(outputs, torch.Tensor) and outputs.numel() == 0) or \
           (isinstance(outputs, list) and not outputs):
            return
    
        self.total_recognition_samples += targets.shape[0]

        if isinstance(outputs, list):
            # predictions: [B] -> torch.stack -> [B, T]
            pred_labels = torch.stack([logits.argmax(dim=1) for logits in outputs], dim=1)

        elif isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                pred_labels = torch.argmax(outputs, dim=2)
            elif outputs.ndim == 2:
                pred_labels = outputs.long()
            else:
                raise TypeError(f"Format not allowed: (dim {outputs.ndim})")
        else:
            raise TypeError(f"Unsupported 'outputs' format (Metrics.update_recognition): {type(outputs)}")
        
        targets_valid = targets.to(pred_labels.device)
        num_valid_predictions = pred_labels.size(0)
        assert num_valid_predictions == targets_valid.size(0) #Check

        for i in range(num_valid_predictions):
            is_full = torch.equal(pred_labels[i], targets_valid[i])
            is_wo_first = torch.equal(pred_labels[i][1:], targets_valid[i][1:])

            if is_full:
                self.correct_plates_full += 1
            if is_wo_first:
                self.correct_chars_wo_first += 1


    def update_detection_time(self, time_spent):
        self.total_time_detection += time_spent
        
    def update_pipeline_time(self, time_spent):
        self.total_time_pipeline += time_spent
    
    def compute(self):
        if self.total_samples == 0:
            return {}
        
        fps_detection = float(self.total_samples / self.total_time_detection) if self.total_time_detection > 0 else 0.0
        fps_pipeline = float(self.total_samples / self.total_time_pipeline) if self.total_time_pipeline > 0 else 0.0

        results = {
            # Nuove metriche FPS
            'FPS_Detection': fps_detection,
            'FPS_Full': fps_pipeline,
            'Mean_IoU': float(np.mean(self.mean_iou)) if self.mean_iou else 0.0,
            'Detection_Accuracy_IoU_0.7': float(100 * self.correct_detections_iou7 / self.total_samples),
            'Plate_Accuracy_Full': float(100 * self.correct_plates_full / self.total_samples),
            'Plate_Accuracy(-Fisrt_Char)': float(100 * self.correct_chars_wo_first / self.total_samples),
        }

        if self.recognition:
            results['Plate_Accuracy_Full_on_Processed'] = float(100 * self.correct_plates_full / self.total_recognition_samples)
            results['Plate_Accuracy(-First)_on_Processed'] = float(100 * self.correct_chars_wo_first / self.total_recognition_samples)

        return {
            k: round(v, 4) if k == 'Mean_IoU' else round(v, 2)
            for k, v in results.items()
        }

class Evaluator:
    def __init__(self, det_model,rec_model, device):
        self.det_model = det_model.to(device)
        self.rec_model = rec_model.to(device)
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
        self.transform_recognition = transform_recognition
        self.transform_detection = transform_detection

        if isinstance(self.det_model, torch.nn.parallel.DistributedDataParallel):
            self.det_model = self.det_model.module
        if isinstance(self.rec_model, torch.nn.parallel.DistributedDataParallel):
            self.rec_model = self.rec_model.module
        
    @torch.no_grad()
    def evaluate(self, dataloader, recognition=False, attack_config=None, debug_prints=False):
        self.det_model.eval()
        self.rec_model.eval()

        metrics = Metrics(task="end_to_end",recognition=recognition)
        is_main_process = not is_initialized() or get_rank() == 0
        desc = "[Attacking]" if attack_config else "[Evaluating]"
        iterator = tqdm(enumerate(dataloader), desc=f"{desc} end-to-end", total=len(dataloader)) if is_main_process else enumerate(dataloader)

        debug_print_count = 0
        max_debug_prints = 20

        for batch_idx, (images, gt_bboxes, plate_labels) in iterator:
            self.start_event.record()
     
            images = images.to(self.device)

            
            # Detection
            if isinstance(self.det_model, YOLOv5):
                gt_bboxes_list = gt_bboxes
                gt_bboxes = torch.stack(gt_bboxes_list, dim=1).to(self.device)

                start_det_event = torch.cuda.Event(enable_timing=True)
                end_det_event = torch.cuda.Event(enable_timing=True)
                start_det_event.record()

                pred_bboxes_norm = self.det_model(images)

                end_det_event.record()
                torch.cuda.synchronize()
                detection_time_spent = start_det_event.elapsed_time(end_det_event) / 1000.0  
                metrics.update_detection_time(detection_time_spent)
                
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

                start_det_event = torch.cuda.Event(enable_timing=True)
                end_det_event = torch.cuda.Event(enable_timing=True)
                start_det_event.record()

                pred_bboxes_norm = self.det_model(images, mode='detection')

                end_det_event.record()
                torch.cuda.synchronize()
                detection_time_spent = start_det_event.elapsed_time(end_det_event) / 1000.0  
                metrics.update_detection_time(detection_time_spent)
                
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
                    labels_tensor = torch.stack(labels_for_rec).to(cropped_batch.device)

                    eval_batch = cropped_batch

                    outputs = torch.empty(0, labels_tensor.size(1), device=labels_tensor.device, dtype=torch.long)
                    if isinstance(self.rec_model, PDLPR):
                        outputs_dirty = self.rec_model(eval_batch)
                        pred_indices = outputs_dirty.argmax(dim=-1)

                        # -------- INIZIO BLOCCO DI DEBUG --------
                        if is_main_process and debug_prints and debug_print_count < max_debug_prints:
                            print("\n" + "="*50)
                            print(f"DEBUG: Dettagli Batch {batch_idx}")
                            print("="*50)
                            
                            for i in range(pred_indices.size(0)):
                                if debug_print_count >= max_debug_prints: break
                                
                                # 1. Ground Truth
                                gt_indices = labels_tensor[i].tolist()
                                gt_text = tokenizer.decode(gt_indices)
                                
                                # 2. Predizione "sporca" (indici grezzi)
                                raw_pred_indices = pred_indices[i].tolist()
                                
                                # 3. Predizione "pulita" (dopo post-processing)
                                try:
                                    first_pad_index = raw_pred_indices.index(0)
                                    cleaned_indices = raw_pred_indices[:first_pad_index]
                                except ValueError:
                                    cleaned_indices = raw_pred_indices
                                
                                # Qui usiamo la logica di pulizia che stai testando.
                                # Se il tuo modello non genera <EOS>, non fare `[:-1]`
                                final_sequence_list = cleaned_indices 
                                final_text = tokenizer.decode(final_sequence_list)
                                
                                print(f"\n--- Esempio {debug_print_count+1} ---")
                                print(f"  Ground Truth: '{gt_text}' -> Indici: {gt_indices}")
                                print(f"  Predizione   (RAW): " + " ".join(map(str, raw_pred_indices)))
                                print(f"  Predizione (Clean): '{final_text}' -> Indici: {final_sequence_list}")
                                
                                debug_print_count += 1
                            
                            if debug_print_count >= max_debug_prints:
                                print("\n[DEBUG] Limite stampe raggiunto. Disattivazione per il resto della valutazione.")
                                debug_prints = False # Disattiva per il resto del ciclo
                        # -------- FINE BLOCCO DI DEBUG --------

                        cleaned_preds_list = []
                        for i in range(pred_indices.size(0)):
                            # Removing padding tokens
                            current_pred_list = pred_indices[i].tolist()
                            try:
                                first_pad_index = current_pred_list.index(0)
                                sequence_without_padding = current_pred_list[:first_pad_index]
                            except ValueError:
                                sequence_without_padding = current_pred_list
       
                            #removing last token <EOS>
                            if len(sequence_without_padding) > 0:
                                final_sequence_list = sequence_without_padding[:-1]
                            else:
                                final_sequence_list = []
                            cleaned_seq_tensor = torch.tensor(final_sequence_list, device=pred_indices.device)

                            padding_needed = labels_tensor.size(1) - len(cleaned_seq_tensor)
                            padded_seq = torch.nn.functional.pad(cleaned_seq_tensor, (0, padding_needed), 'constant', 0)
                            cleaned_preds_list.append(padded_seq)
                            
                        if cleaned_preds_list:
                            outputs = torch.stack(cleaned_preds_list)
                    else:
                        outputs = self.rec_model(eval_batch, mode='recognition')

                    metrics.update_recognition(outputs, labels_tensor)
                
            self.end_event.record()
            torch.cuda.synchronize()
            pipeline_time_spent = self.start_event.elapsed_time(self.end_event) / 1000.0
            metrics.update_pipeline_time(pipeline_time_spent)

        if is_initialized():
            barrier()

        if is_main_process:
            return metrics.compute()
        else:
            return None
 
class Trainer:
    def __init__(self, model, task, device, lr=1e-3, num_classes_list=None):
        self.model = model.to(device)
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[device], find_unused_parameters=True)
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
    
    def train(self, dataloader=None, val_loader=None, epochs=10, batch_size=50, optimizer="Adam",lr0=1e-3,lrf=1e-5, cos_lr=True,project="runs/train", name="lp_detection", cache="ram", adversarial=False, epsilon=0.03):
        if self.model_type == "yolov5":
            return self._train_yolov5(epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr0=lr0, lrf=lrf, cos_lr=cos_lr, project=project, name=name, cache=cache)
        elif self.model_type == "pdlpr":
            return self._train_pdlpr(dataloader, val_loader, epochs)
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

    def _train_pdlpr(self, dataloader, val_dataloader, epochs):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

        train_losses = []
        val_losses = []

        lambda_reg = 1e-7  # coefficiente di regolarizzazione (da testare e calibrare)

        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [PDLPR Train]", unit="batch")
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                

                # Abilita la derivazione rispetto agli input
                images.requires_grad = True

                self.optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    output = self.model(images)
                    output = output.permute(0, 2, 1)  # [B, SeqLen, C] -> [B, C, SeqLen]
                    loss_ce = loss_fn(output, targets)

                # Calcolo del gradiente della loss rispetto all'input
                grad_input = torch.autograd.grad(loss_ce, images, create_graph=True, retain_graph=True)[0]
                grad_loss = grad_input.pow(2).mean() 

                # Loss totale con regolarizzazione
                loss_total = loss_ce + lambda_reg * grad_loss

                scaler.scale(loss_total).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss_ce.item()
                pbar.set_postfix({"batch_loss": loss_ce.item()})

            avg_train_loss = running_loss / len(dataloader)
            train_losses.append(avg_train_loss)
            disp(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [PDLPR Val]", unit="batch"):
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    with autocast(device_type="cuda"):
                        output = self.model(images)
                        output = output.permute(0, 2, 1)
                        loss = loss_fn(output, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            disp(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {avg_val_loss:.4f}")

            torch.save(self.model.state_dict(), f"src/PDLPR/weights/newtrain/pdlpr_epoch{epoch+1}.pth")

            self.model.train()

        torch.save(self.model.state_dict(), "src/PDLPR/weights/newtrain/pdlpr_final_new.pth")
        self.plot_epoch_losses(train_losses, val_losses, "PDLPR Loss", "src/PDLPR/logs/newtrain")
        return self.model



# MODELS #

def collate_fn_end_to_end_filter(batch):
    batch = [sample for sample in batch if sample[1] is not None]
    if not batch:
        return torch.empty(0, 3, 640, 640), torch.empty(0, 4), torch.empty(0, seq_len)
    return torch.utils.data.dataloader.default_collate(batch)

def collate_fn_recognition(batch):
    batch = [sample for sample in batch if sample[1] is not None and len(sample[1]) > 0]
    if not batch:
        return torch.empty(0, 3, 64, 128), torch.empty(0, seq_len, dtype=torch.long)

    images, plate_codes = zip(*batch)
    images = torch.stack(images)
    
    padded_codes = []
    for code in plate_codes:
        code = code[:seq_len]
        padding_needed = seq_len - len(code)
        padded_code = F.pad(code, (0, padding_needed), 'constant', 0)
        padded_codes.append(padded_code)
    targets = torch.stack(padded_codes).long() 
    
    return images, targets

def collate_fn_for_attack_generation(batch):
    batch = [sample for sample in batch if sample[1] is not None and sample[3] is not None]
    if not batch:
        return torch.empty(0), torch.empty(0), [], []

    images, plate_codes, filenames, bboxes_abs = zip(*batch)
    plate_codes = torch.stack(plate_codes, 0)
    
    return images, plate_codes, list(filenames), list(bboxes_abs)

# --- IGFE ---
class FocusStructure(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cnn_block1 = CNNBlock(in_channels, out_channels)
        self.cnn_block2 = CNNBlock(out_channels, out_channels)
        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        identity = self.identity(x)
        out = self.cnn_block1(x)
        out = self.cnn_block2(out)
        return out + identity

class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.conv(x)
        return x

class IGFE(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.focus = FocusStructure()
        self.layer1 = ResBlock(4 * in_channels, base_channels)
        self.layer2 = ResBlock(base_channels, base_channels)
        self.down1 = ConvDownSampling(base_channels, base_channels, stride=2)
        self.layer3 = ResBlock(base_channels, base_channels)
        self.layer4 = ResBlock(base_channels, base_channels)
        self.down2 = ConvDownSampling(base_channels, base_channels, stride=2)
    def forward(self, x):
        x = self.focus(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.down1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.down2(x)
        return x
    
# --- Encoder ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")
        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (d_model // 2)))
        pe[0::4, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::4, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[2::4, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[3::4, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]

class EncoderModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(d_model, height, width)
        self.cnn_block1 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.cnn_block2 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.add_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x.clone()
        x = self.pos_enc(x)
        x = self.cnn_block1(x)
        B, C, H, W = x.shape
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)
        attn_out, _ = self.mha(x_, x_, x_)
        x = attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        x = self.cnn_block2(x)
        out = residual + x
        out = out.permute(0, 2, 3, 1)
        out = self.add_norm(out)
        out = out.permute(0, 3, 1, 2)
        return out

class Encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderModule(d_model, nhead, height, width) for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Decoder ---
class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    def forward(self, x, sublayer_out):
        return self.norm(x + sublayer_out)

# SOSTITUZIONE FINALE PER DecodingModule
class DecodingModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16):
        super().__init__()
        # I layer dell'init rimangono IDENTICI a prima, quindi i pesi sono compatibili.
        self.pos_enc = PositionalEncoding2D(d_model, height, width)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.cross_cnn1 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.cross_cnn2 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(d_model, d_model * 4, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model * 4, d_model, kernel_size=1),
        )
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, x, encoder_out):
        # --- Preparazione Input ---
        # Applica l'encoding posizionale all'input x
        x_with_pos = self.pos_enc(x)

        B, C, H, W = x_with_pos.shape
        # Prepara la vista per la self-attention: [Seq, Batch, Feat]
        qkv = x_with_pos.permute(2, 3, 0, 1).reshape(H * W, B, C)

        # --- 1. Self-Attention ---
        # Usa la stessa vista per query, key, value
        self_attn_res, _ = self.self_attn(qkv, qkv, qkv)
        # Add & Norm: il residual è qkv, il risultato è self_attn_res
        # self_attn_out ha forma [Seq, Batch, Feat]
        self_attn_out = self.addnorm1(qkv.permute(1,0,2), self_attn_res.permute(1,0,2)).permute(1,0,2)

        # --- 2. Cross-Attention ---
        # Prepara l'output dell'encoder
        enc = self.cross_cnn1(encoder_out)
        enc = self.cross_cnn2(enc)
        B_enc, C_enc, H_enc, W_enc = enc.shape
        # Vista per key e value dall'encoder: [Seq_enc, Batch, Feat]
        kv_enc = enc.permute(2, 3, 0, 1).reshape(H_enc * W_enc, B_enc, C_enc)

        # La query (q) viene dal risultato della self-attention (self_attn_out)
        # Key (k) e Value (v) vengono dall'encoder (kv_enc)
        cross_attn_res, _ = self.cross_attn(query=self_attn_out, key=kv_enc, value=kv_enc)
        # Add & Norm: il residual è self_attn_out, il risultato è cross_attn_res
        cross_attn_out = self.addnorm2(self_attn_out.permute(1,0,2), cross_attn_res.permute(1,0,2)).permute(1,0,2)

        # --- 3. Feed Forward ---
        # Riporta al formato immagine per le Conv2d: [Batch, Feat, H, W]
        ff_input = cross_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        ff_res = self.feed_forward(ff_input)
        
        # Add & Norm: il residual è ff_input, il risultato è ff_res
        # L'input per addnorm3 deve essere [B, ..., C]
        out = self.addnorm3(ff_input.permute(0,2,3,1).reshape(B, -1, C), ff_res.permute(0,2,3,1).reshape(B, -1, C))

        # Riporta al formato finale richiesto
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

class Decoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16, num_layers=3, num_classes=68, seq_len=8):
        super().__init__()
        self.layers = nn.ModuleList([
            DecodingModule(d_model=d_model, nhead=nhead, height=height, width=width)
            for _ in range(num_layers)
        ])
        self.seq_len = seq_len
        self.classifier = nn.Linear(d_model, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, seq_len))  # (B, C, 1, seq_len)
    def forward(self, x, encoder_out):
        for layer in self.layers:
            x = layer(x, encoder_out)
        x = self.pool(x)  # (B, C, 1, seq_len)
        x = x.squeeze(2)  # (B, C, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, C)
        logits = self.classifier(x)  # (B, seq_len, num_classes)
        return logits

## ----- PDLPR MODEL ----- ##
class PDLPR(nn.Module):
    def __init__(self,
                 model_path=None,
                 in_channels=3,
                 base_channels=256,
                 encoder_d_model=256,
                 encoder_nhead=8,
                 encoder_height=16,
                 encoder_width=16,
                 decoder_num_layers=3,
                 num_classes=68,
                 seq_len=8):
        super().__init__()
        
        self.model_type = "pdlpr"
        
        self.igfe = IGFE(in_channels, base_channels)
        
        self.pool = nn.AdaptiveAvgPool2d((encoder_height, encoder_width))
        
        self.encoder = Encoder(
            d_model=encoder_d_model, 
            nhead=encoder_nhead, 
            height=encoder_height, 
            width=encoder_width
        )
        
        self.decoder = Decoder(
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            height=encoder_height,
            width=encoder_width,
            num_layers=decoder_num_layers,
            num_classes=num_classes,
            seq_len=seq_len
        )

        state_dict = torch.load(model_path, map_location=DEVICE)
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
        else:
            self.load_state_dict(state_dict)
        
    def forward(self, x):
        x = self.igfe(x)
        x = self.pool(x)
        x = self.encoder(x)
        decoder_input = torch.zeros_like(x)
        x = self.decoder(decoder_input, x)
        return x
    
class YOLOv5(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model = attempt_load(model_path, device=device) 

    def forward(self, img):
        with torch.no_grad():
            preds = self.model(img)[0]
            preds = non_max_suppression(prediction=preds)

            bboxes = []
            for pred in preds:
                if pred is not None and len(pred) > 0:
                    # BBox with highest confidence
                    best = pred[torch.argmax(pred[:, 4])]
                    bbox = best[:4]  # [x1, y1, x2, y2]
                else:
                    bbox = torch.tensor([0, 0, 0, 0], dtype=torch.float32, device=img.device)

                # Normalizing in 0,1
                h, w = img.shape[2:] #640x640
                norm_bbox = bbox / torch.tensor([w, h, w, h], device=img.device)
                bboxes.append(norm_bbox)

            return torch.stack(bboxes)
        return preds



def generate_attacked_dataset(model, dataset, save_dir, epsilon=0.01, batch_size=32, device="cuda"):
    """
    Applica FGSM alla regione della targa e la incolla sull'immagine originale.
    Salva l'immagine attaccata con il nome del file originale.
    """
    os.makedirs(save_dir, exist_ok=True)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # DataLoader con il sampler
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_for_attack_generation,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    transform_rec = T.Compose([T.Resize((64, 128)), T.ToTensor()])
    to_pil = T.ToPILImage()

    for param in model.parameters():
        param.requires_grad = False

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Generating Adversarial Dataset (Total)", total=len(dataloader))
    else:
        pbar = dataloader

    for _, targets, filenames, bboxes_abs in pbar:
        
        # --- 1. Prepara il batch di sole targhe ritagliate ---
        crops_for_attack = []
        valid_indices = [] # Tiene traccia degli indici validi nel batch
        
        for i, filename in enumerate(filenames):
            try:
                original_path = os.path.join(dataset.img_dir, filename)
                original_pil_img = Image.open(original_path).convert("RGB")
                
                # Ritaglia la targa usando le coordinate assolute
                x1, y1, x2, y2 = bboxes_abs[i]
                cropped_pil = original_pil_img.crop((x1, y1, x2, y2))
                
                # Applica la trasformazione che il modello si aspetta
                cropped_tensor = transform_rec(cropped_pil)
                crops_for_attack.append(cropped_tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                continue
        
        if not crops_for_attack:
            continue

        # Crea il batch di sole targhe e le relative etichette
        crops_batch = torch.stack(crops_for_attack).to(device).requires_grad_(True)
        targets_batch = targets[valid_indices].to(device)

        # --- 2. Esegui l'attacco FGSM sulla targa ---
        model.zero_grad()
        outputs = model(crops_batch)
        outputs = outputs.permute(0, 2, 1) # [B, SeqLen, C] -> [B, C, SeqLen]
        loss = loss_fn(outputs, targets_batch)
        loss.backward()

        data_grad = crops_batch.grad.data
        perturbed_crops_batch = crops_batch + epsilon * data_grad.sign()
        perturbed_crops_batch = torch.clamp(perturbed_crops_batch, 0, 1)

        # --- 3. Incolla le targhe attaccate e salva ---
        for i, batch_idx in enumerate(valid_indices):
            filename = filenames[batch_idx]
            bbox_abs = bboxes_abs[batch_idx]
            
            # Carica di nuovo l'immagine originale pulita
            original_path = os.path.join(dataset.img_dir, filename)
            final_image = Image.open(original_path).convert("RGB")
            
            # Prendi la targa perturbata
            perturbed_crop_tensor = perturbed_crops_batch[i].cpu().detach()
            perturbed_crop_pil = to_pil(perturbed_crop_tensor)
            
            # Ri-dimensiona la targa attaccata alla sua dimensione originale
            original_width = bbox_abs[2] - bbox_abs[0]
            original_height = bbox_abs[3] - bbox_abs[1]
            perturbed_crop_pil = perturbed_crop_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            
            # Incolla la targa attaccata sull'immagine pulita
            final_image.paste(perturbed_crop_pil, (bbox_abs[0], bbox_abs[1]))
            
            # Salva con il nome del file originale
            save_path = os.path.join(save_dir, filename)
            final_image.save(save_path)
 
 
def main():

    pdlpr_orig_path = "src/PDLPR/weights/pdlpr_final.pth"
    pdlpr_model_path = "src/PDLPR/weights/newtrain/pdlpr_final_new.pth"
    yolo_model_path = "src/YOLO/runs/train/weights/best.pt"
    yolov5_model = YOLOv5(model_path=yolo_model_path, device=DEVICE)
    pdlpr_model = PDLPR(
        model_path=pdlpr_orig_path, #Check pdlpr_orig_path
        in_channels=3,
        base_channels=256,
        encoder_d_model=256,
        encoder_nhead=4,
        encoder_height=16,
        encoder_width=16,
        decoder_num_layers=2,
        num_classes=num_classes,
        seq_len=seq_len
    )
    pdlpr_original = PDLPR(
        model_path=pdlpr_orig_path,
        in_channels=3,
        base_channels=256,
        encoder_d_model=256,
        encoder_nhead=4,
        encoder_height=16,
        encoder_width=16,
        decoder_num_layers=2,
        num_classes=num_classes,
        seq_len=seq_len
    )
    num_classes_list = [len(PROVINCES), len(ALPHABETS)] + [len(ADS)] * 5 
    
    transform_rec = T.Compose([T.Resize((64, 128)), T.ToTensor()])

    val_dataset = CCPDDataset(VALIDATION_PATH, transform=transform_rec, task='recognition', model="pdlpr", max_samples=5000)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, collate_fn=collate_fn_recognition) # USARE COLLATe
    
    TEST_PATH = os.path.join(os.environ["SCRATCH"], "dataset", "CCPD_YOLO", "ccpd_challenge", "images", "test")
    test_dataset = CCPDDataset(TEST_PATH, transform=transform_yolo, task='end_to_end', model="yolov5", max_samples=2000)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_end_to_end_filter, num_workers=8)


    # Generation attacked dataset
    if False:
        gen_dataset = CCPDDataset(TEST_PATH, transform=None, task='end_to_end', model="yolov5", max_samples=2000, for_attack_generation=True)
        generate_attacked_dataset(
            model=pdlpr_original,
            dataset=gen_dataset,
            save_dir=os.path.join(os.environ["SCRATCH"], "dataset", "CCPD_YOLO", "ccpd_challenge_att", "images", "test"),
            epsilon=0.01,
            batch_size=64,
            device=DEVICE
        )

    if False:
        gen_dataset = CCPDDataset(TRAINING_PATH, transform=None, task='end_to_end', model="yolov5", max_samples=80000, for_attack_generation=True)
        generate_attacked_dataset(
            model=pdlpr_original,
            dataset=gen_dataset,
            save_dir=os.path.join(os.environ["SCRATCH"], "dataset", "CCPD_YOLO", "ccpd_base_att", "images", "train"),
            epsilon=0.01,
            batch_size=64,
            device=DEVICE
        )
        barrier()
        disp("Dataset generation complete on all processes.")

    if True:
        dataset_attacked_path = os.path.join(os.environ["SCRATCH"], "dataset", "CCPD_YOLO", "ccpd_base_att", "images", "train")
        train_dataset_base = CCPDDataset(TRAINING_PATH, transform=transform_rec, task='recognition', model="pdlpr")
        train_dataset_attacked = CCPDDataset(dataset_attacked_path, transform=transform_rec, task='recognition', model="pdlpr")

        print("Combining datasets...")
        combined_dataset = ConcatDataset([train_dataset_base, train_dataset_attacked])
        print(f"Total images in combined dataset: {len(combined_dataset)}")

        train_sampler = DistributedSampler(combined_dataset, shuffle=True)
        train_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False, sampler=train_sampler, num_workers=4, collate_fn=collate_fn_recognition)


        trainer = Trainer(pdlpr_model, task='recognition', num_classes_list=num_classes_list, device=DEVICE, lr=5e-5) # LR più basso per fine-tuning
        pdlpr_model = trainer.train(dataloader=train_loader, val_loader=val_loader, epochs=5, adversarial=False)

    evaluator_original = Evaluator(det_model=yolov5_model, rec_model=pdlpr_original, device=DEVICE)
    evaluator_trained = Evaluator(det_model=yolov5_model, rec_model=pdlpr_model, device=DEVICE)

    
    disp("\n--- Valutazione ---")
    metrics_orig_clean = evaluator_original.evaluate(test_loader, recognition=True, debug_prints=True)
    disp(f"Modello Originale (Clean): {metrics_orig_clean}")

    metrics_robust_clean = evaluator_trained.evaluate(test_loader, recognition=True, debug_prints=True)
    disp(f"Modello Trained (Clean):   {metrics_robust_clean}")


    TEST_PATH_ATT = os.path.join(os.environ["SCRATCH"], "dataset", "CCPD_YOLO", "ccpd_challenge_att", "images", "test")
    test_dataset_att = CCPDDataset(TEST_PATH_ATT, transform=transform_yolo, task='end_to_end', model="yolov5", max_samples=2000)
    test_loader_att = DataLoader(test_dataset_att, batch_size=128, shuffle=False, collate_fn=collate_fn_end_to_end_filter, num_workers=8)

    disp(f"\n--- Valutazione su DATI ATTACCATI ---")
    
    metrics_orig_attacked = evaluator_original.evaluate(test_loader_att, recognition=True, debug_prints=True)
    disp(f"Modello Originale (Attacked): {metrics_orig_attacked}")
    
    metrics_robust_attacked = evaluator_trained.evaluate(test_loader_att, recognition=True, debug_prints=True)
    disp(f"Modello Trained (Attacked):   {metrics_robust_attacked}")

    return

if __name__ == "__main__":
    main()
    