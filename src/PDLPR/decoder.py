import torch.nn as nn
from src.PDLPR.encoder import PositionalEncoding2D
from src.PDLPR.igfe import CNNBlock

class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, sublayer_out):
        # x and sublayer_out shape (B, SeqLen, Dim)
        return self.norm(x + sublayer_out)

class DecodingModule(nn.Module):
    def __init__(self, d_model=1024, nhead=8, height=16, width=16):
        super(DecodingModule, self).__init__()

        # Positional encoding per decoder input
        self.pos_enc = PositionalEncoding2D(d_model, height, width)

        # Self Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Cross Attention
        self.cross_cnn1 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.cross_cnn2 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Conv2d(d_model, d_model * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(d_model * 4, d_model, kernel_size=1),
        )

        # Add & Norm layers
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, x, encoder_out):
        # x shape: (B, C=d_model, H, W)
        # encoder_out shape: (B, C=d_model, H_enc, W_enc)

        residual = x.clone()
        x = self.pos_enc(x)

        B, C, H, W = x.shape
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)  # (SeqLen, Batch, EmbDim)

        # Self Attention
        self_attn_out, _ = self.self_attn(x_, x_, x_)
        self_attn_out = self.addnorm1(x_.permute(1, 0, 2), self_attn_out.permute(1, 0, 2))  # (B, SeqLen, C)

        # Torna a (SeqLen, B, C)
        self_attn_out = self_attn_out.permute(1, 0, 2)

        # Reshape in (B, C, H, W)
        x = self_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)

        # Conv per cross attention
        enc = self.cross_cnn1(encoder_out)
        enc = self.cross_cnn2(enc)

        B_enc, C_enc, H_enc, W_enc = enc.shape
        enc_ = enc.permute(2, 3, 0, 1).reshape(H_enc*W_enc, B_enc, C_enc)  # (SeqLen_enc, B, C)

        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)  # (SeqLen_dec, B, C)

        cross_attn_out, _ = self.cross_attn(x_, enc_, enc_)
        cross_attn_out = self.addnorm2(x_.permute(1, 0, 2), cross_attn_out.permute(1, 0, 2))  # (B, SeqLen, C)

        # Ritrasformo a (SeqLen, B, C)
        cross_attn_out = cross_attn_out.permute(1, 0, 2)
        x = cross_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)  # (B, C, H, W)

        # Feed Forward
        ff_out = self.feed_forward(x)

        # Add & Norm 
        out = self.addnorm3(x.permute(0, 2, 3, 1).reshape(B, -1, C), ff_out.permute(0, 2, 3, 1).reshape(B, -1, C))
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out





class Decoder(nn.Module):
    def __init__(self, d_model=1024, nhead=8, height=16, width=16, num_layers=3):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecodingModule(d_model=d_model, nhead=nhead, height=height, width=width)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_out):
        # x: (B, C=d_model, H, W)
        # encoder_out: (B, C=d_model, H_enc, W_enc)
        for layer in self.layers:
            x = layer(x, encoder_out)
        return x