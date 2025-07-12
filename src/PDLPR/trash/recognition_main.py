import os
import random
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from src.PDLPR.PDLPR import PDLPR  # Assumo tu abbia già implementato questo modulo


# ====== COSTANTI E UTILS ======

# Path dataset
image_folder_1 = r"C:\Users\loreb\Desktop\CV\dataset\dataset_minor"

# Liste di mapping targa CCPD
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


def decode_plate(plate_code):
    """Decodifica la targa da codici interi a stringa"""
    try:
        province = provinces[plate_code[0]]
        letter = alphabets[plate_code[1]]
        tail = ''.join(ads[i] for i in plate_code[2:])
        return province + letter + tail
    except IndexError:
        return "INVALID_CODE"


def parse_filename(filename):
    """Estrae info da filename CCPD con struttura nota"""
    parts = filename[:-4].split('-')
    if len(parts) != 7:
        raise ValueError("Formato filename errato")

    area = float(parts[0])
    tilt = tuple(map(int, parts[1].split('_')))
    bbox = [tuple(map(int, p.split('&'))) for p in parts[2].split('_')]
    points = [tuple(map(int, p.split('&'))) for p in parts[3].split('_')]
    plate_code = list(map(int, parts[4].split('_')))
    brightness = int(parts[5])
    blurriness = int(parts[6])

    plate_text = decode_plate(plate_code)

    return {
        "area": area,
        "tilt": tilt,
        "bbox": bbox,
        "points": points,
        "plate_code": plate_code,
        "plate_text": plate_text,
        "brightness": brightness,
        "blurriness": blurriness
    }


def show_random_ccpd_image(folder_path):
    """Mostra un'immagine casuale con info della targa"""
    all_images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    if not all_images:
        print("Nessuna immagine trovata nella cartella.")
        return

    filename = random.choice(all_images)
    info = parse_filename(filename)

    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path)

    plt.imshow(img)
    plt.title(f"Targa: {info['plate_text']}\nFile: {filename}")
    plt.axis('off')
    plt.show()

    print("Informazioni CCPD:")
    for k, v in info.items():
        print(f"{k}: {v}")


# ====== DATASET E TOKENIZER ======

default_transform = transforms.Compose([
    transforms.Resize((720, 1160)),  # Altezza, Larghezza
    transforms.ToTensor()
])


class CCPDPlateDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform if transform else default_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        filepath = os.path.join(self.image_folder, filename)

        image = Image.open(filepath).convert("RGB")

        info = parse_filename(filename)
        (x1, y1), (x2, y2) = info["bbox"]

        plate_image = image.crop((x1, y1, x2, y2))
        plate_tensor = self.transform(plate_image)

        label_text = info["plate_text"]

        return plate_tensor, label_text


class SimplePlateTokenizer:
    def __init__(self, charset):
        self.char2idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 = PAD
        self.char2idx['<PAD>'] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]  # 0 per PAD o char ignoto

    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '') for i in indices if i != 0])

    def vocab_size(self):
        return len(self.char2idx)


# Charset + tokenizer
charset = provinces + alphabets + [str(i) for i in range(10)]
tokenizer = SimplePlateTokenizer(charset)


def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)

    token_seqs = [torch.tensor(tokenizer.encode(t)) for t in texts]
    target_lens = torch.tensor([len(seq) for seq in token_seqs])
    padded_targets = pad_sequence(token_seqs, batch_first=True, padding_value=0)  # 0 = PAD

    return images, padded_targets, target_lens


# ====== MODELLO, LOSS, OTTIMIZZATORE ======

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PDLPR(
    in_channels=3,
    base_channels=512,
    encoder_d_model=512,
    encoder_nhead=8,
    encoder_height=16,
    encoder_width=16,
    decoder_num_layers=3
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignora PAD token


# ====== CARICAMENTO DATASET E DATALOADER ======

dataset = CCPDPlateDataset(image_folder=image_folder_1)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


# ====== TRAINING LOOP ======

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets, target_lens in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(images)  # Output shape: [B, T, V]

        # Per CrossEntropyLoss, shape richiesta: [B, V, T]
        output = output.permute(0, 2, 1)

        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Salvataggio checkpoint ad ogni epoca
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, "pdlpr_checkpoint.pth")
