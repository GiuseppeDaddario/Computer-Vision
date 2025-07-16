import torch
import torch.nn as nn
import torch.nn.functional as F


from src.PDLPR.igfe import IGFE
from src.PDLPR.encoder import Encoder
from src.PDLPR.decoder import Decoder

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