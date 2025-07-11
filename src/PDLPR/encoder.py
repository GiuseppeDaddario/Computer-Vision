import torch
import torch.nn as nn
import math

from src.PDLPR.igfe import CNNBlock





class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")

        self.d_model = d_model
        self.height = height
        self.width = width

        # Create positional encoding once in constructor
        pe = torch.zeros(d_model, height, width)  # shape (C, H, W)

        # Position indices
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)  # (H, W)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)  # (H, W)

        div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(math.log(10000.0) / (d_model // 2)))

        # sin to even indices in dim 0, cos to odd indices
        # First half channels encode Y (height)
        pe[0::4, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::4, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        # Second half channels encode X (width)
        pe[2::4, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[3::4, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))

        # buffer
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, C, H, W)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]
        return x





class AddNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x, sublayer_out):
        # x, sublayer_out shape= (B, SeqLen, Dim)
        return self.norm(x + sublayer_out)
    




import torch
import torch.nn as nn

class EncoderModule(nn.Module):
    def __init__(self, d_model=1024, nhead=8, height=16, width=16):
        super(EncoderModule, self).__init__()

        self.pos_enc = PositionalEncoding2D(d_model, height, width)
        self.cnn_block1 = CNNBlock(in_channels=d_model, out_channels=d_model, stride=1, kernel_size=1, padding=0)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.cnn_block2 = CNNBlock(in_channels=d_model, out_channels=d_model, stride=1, kernel_size=1, padding=0)
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
    def __init__(self, d_model=1024, nhead=8, height=16, width=16, num_layers=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderModule(d_model, nhead, height, width) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x