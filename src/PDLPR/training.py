import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from torch.nn.utils.rnn import pad_sequence

from src.PDLPR.preprocessing import crop_plate
from src.PDLPR.PDLPR import PDLPR


import matplotlib.pyplot as plt
from src.PDLPR.augmentation import FullRobustAugmentation

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Costanti CCPD ---

# --- CCPD CHARSET FIXED ---
provinces = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
    "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
    "青", "宁", "新", "警", "学", "O"
]
alphabets = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O'
]
ads = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'
]


# Charset: province + alphabets (solo quelli non già in province) + numeri (0-9), senza duplicati
charset = provinces + [c for c in alphabets if c not in provinces] + [str(i) for i in range(10)]
charset = list(dict.fromkeys(charset))  # rimuove duplicati mantenendo l'ordine

def decode_plate(plate_code):
    try:
        province = provinces[plate_code[0]]
        letter = alphabets[plate_code[1]]
        tail = ''.join(ads[i] for i in plate_code[2:])
        return province + letter + tail
    except Exception:
        return "INVALID"

def parse_filename(filename):
    parts = filename[:-4].split('-')
    plate_code = list(map(int, parts[4].split('_')))
    return decode_plate(plate_code)

class SimplePlateTokenizer:
    def __init__(self, charset):
        self.char2idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 = PAD
        self.char2idx['<PAD>'] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}
    def encode(self, text):
        # DEBUG: segnala caratteri non nel charset
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
seq_len = 8  # lunghezza massima targa CCPD



# --- Dataset CCPD ---
class CCPDPlateDataset(Dataset):
    def __init__(self, image_folder, transform=None, max_len=8):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform if transform else FullRobustAugmentation()


        self.max_len = max_len
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, filename)
        image = Image.open(img_path).convert("RGB")
        image = crop_plate(img_path)  # Crop the plate from the image
        image = self.transform(image)
        label_text = parse_filename(filename)
        return image, label_text


def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)
    token_seqs = [torch.tensor(tokenizer.encode(t)[:seq_len] + [0]*(seq_len-len(t))) for t in texts]
    targets = torch.stack(token_seqs)  # [B, seq_len]
    # DEBUG: controlla range target
    if (targets >= num_classes).any() or (targets < 0).any():
        print("[ERROR] Target fuori range! Ecco alcune label e codifiche:")
        for t in texts:
            print("Label:", t, "Encoded:", tokenizer.encode(t))
        print("Target tensor:", targets)
        print("num_classes:", num_classes)
        raise ValueError("Target fuori range per CrossEntropyLoss!")
    return images, targets

from torch.utils.data import random_split

def PDLPR_training(image_folder, num_epochs, batch_size=32, rank=0, world_size=1, device=torch.device('cpu')):


    if rank == 0:
        os.makedirs("src/PDLPR/weights/newtrain", exist_ok=True)
        os.makedirs("src/PDLPR/logs/newtrain", exist_ok=True)
    # --- Training setup ---
    #image_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\CCPD2019\ccpd_base" 
    batch_size = batch_size
    dataset = CCPDPlateDataset(image_folder)

    # Suddividi il dataset in 80% train e 20% val
    train_size = int(0.8 * len(dataset)) #qua dovrei rimettere 0.8
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Distributed Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDLPR(
        in_channels=3,
        base_channels=512, #dim igfe
        encoder_d_model=512, # dim encoder-decoder
        encoder_nhead=4,
        encoder_height=18,
        encoder_width=6,
        decoder_num_layers=2,
        num_classes=num_classes,
        seq_len=seq_len
    ).to(device)

    model = DDP(model, device_ids=[device.index], output_device=device.index)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler(device = "cuda")

    

    try:
        from tqdm import tqdm
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm

    train_losses = []
    val_losses = []


    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(images)
                output = output.permute(0, 2, 1)
                loss = loss_fn(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({"batch_loss": loss.item()})
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch"):
                images = images.to(device)
                targets = targets.to(device)
                with autocast(device_type="cuda"):
                    output = model(images)
                    output = output.permute(0, 2, 1)
                    loss = loss_fn(output, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}")

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}")
            torch.save(model.module.state_dict(), f"src/PDLPR/weights/newtrain/pdlpr_epoch{epoch+1}.pth")

    if rank == 0:
        torch.save(model.module.state_dict(), "src/PDLPR/weights/newtrain/pdlpr_final.pth")

        # Plot loss solo rank 0
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("src/PDLPR/logs/newtrain/loss_plot.png")
        plt.close()
        print("Salvato grafico delle loss in 'src/PDLPR/logs/newtrain/loss_plot.png'")

    
        

  

