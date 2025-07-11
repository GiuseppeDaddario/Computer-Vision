import torch
import torch.nn as nn
from src.PDLPR.igfe import IGFE
from src.PDLPR.encoder import Encoder
from src.PDLPR.decoder import Decoder

import torch.nn.functional as F


class PDLPR(nn.Module):
    def __init__(self, in_channels, base_channels, encoder_d_model, encoder_nhead, encoder_height, encoder_width):
        super(PDLPR, self).__init__()
        self.igfe = IGFE(in_channels, base_channels)
        self.pool = nn.AdaptiveAvgPool2d((encoder_height, encoder_width))
        self.encoder = Encoder(d_model=encoder_d_model, nhead=encoder_nhead, height=encoder_height, width=encoder_width)


        self.decoder = Decoder(  
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            height=encoder_height,
            width=encoder_width,
            num_layers=3
        )
    
    def forward(self, x):
        x = self.igfe(x)
        x = self.pool(x)           
        x = self.encoder(x)
        decoder_input = torch.zeros_like(x)

        x = self.decoder(decoder_input, x)

        return x
    



