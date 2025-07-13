import torch
import torch.nn as nn
import torch.nn.functional as F
from src.PDLPR.igfe import CNNBlock
from src.PDLPR.encoder import PositionalEncoding2D

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