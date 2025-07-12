import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DecodingModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16):
        super().__init__()
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
        x = self.pos_enc(x)
        B, C, H, W = x.shape
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)
        self_attn_out, _ = self.self_attn(x_, x_, x_)
        self_attn_out = self.addnorm1(x_.permute(1, 0, 2), self_attn_out.permute(1, 0, 2))
        self_attn_out = self_attn_out.permute(1, 0, 2)
        x = self_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        enc = self.cross_cnn1(encoder_out)
        enc = self.cross_cnn2(enc)
        B_enc, C_enc, H_enc, W_enc = enc.shape
        enc_ = enc.permute(2, 3, 0, 1).reshape(H_enc*W_enc, B_enc, C_enc)
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)
        cross_attn_out, _ = self.cross_attn(x_, enc_, enc_)
        cross_attn_out = self.addnorm2(x_.permute(1, 0, 2), cross_attn_out.permute(1, 0, 2))
        cross_attn_out = cross_attn_out.permute(1, 0, 2)
        x = cross_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        ff_out = self.feed_forward(x)
        out = self.addnorm3(x.permute(0, 2, 3, 1).reshape(B, -1, C), ff_out.permute(0, 2, 3, 1).reshape(B, -1, C))
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

# --- PDLPR Model ---
class PDLPR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=512,
                 encoder_d_model=512,
                 encoder_nhead=8,
                 encoder_height=16,
                 encoder_width=16,
                 decoder_num_layers=3,
                 num_classes=68,
                 seq_len=8):
        super().__init__()
        self.igfe = IGFE(in_channels, base_channels)
        self.pool = nn.AdaptiveAvgPool2d((encoder_height, encoder_width))
        self.encoder = Encoder(d_model=encoder_d_model, nhead=encoder_nhead, height=encoder_height, width=encoder_width)
        self.decoder = Decoder(
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            height=encoder_height,
            width=encoder_width,
            num_layers=decoder_num_layers,
            num_classes=num_classes,
            seq_len=seq_len
        )
    def forward(self, x):
        x = self.igfe(x)
        x = self.pool(x)
        x = self.encoder(x)
        decoder_input = torch.zeros_like(x)
        x = self.decoder(decoder_input, x)
        return x
    





import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from PIL import Image
#from src.PDLPR.PDLPR import PDLPR

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
print("[INFO] Charset:", charset)
print("[INFO] num_classes:", num_classes)

# --- Dataset CCPD ---
class CCPDPlateDataset(Dataset):
    def __init__(self, image_folder, transform=None, max_len=8):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((48, 144)),
            transforms.ToTensor()
        ])
        self.max_len = max_len
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, filename)
        image = Image.open(img_path).convert("RGB")
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

# --- Training setup ---
image_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\dataset_minor" 
batch_size = 8
dataset = CCPDPlateDataset(image_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PDLPR(
    in_channels=3,
    base_channels=256,
    encoder_d_model=256,
    encoder_nhead=4,
    encoder_height=16,
    encoder_width=16,
    decoder_num_layers=2,
    num_classes=num_classes,
    seq_len=seq_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# --- Training loop ---
num_epochs = 10

# --- Progress bar ---
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)  # [B, seq_len]
        optimizer.zero_grad()
        output = model(images)  # [B, seq_len, num_classes]
        output = output.permute(0, 2, 1)  # [B, num_classes, seq_len]
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": loss.item()})
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    # torch.save(model.state_dict(), "pdlpr_ccpd_checkpoint.pth")