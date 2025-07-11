import torch
import torch.nn as nn
import torch.nn.functional as F



class FocusStructure(nn.Module):
    def __init__(self):
        super(FocusStructure, self).__init__()

    def forward(self, x):
        # x shape: (B, C, H, W)
        # Slice in 4 e concat su canale
        return torch.cat([
            x[..., ::2, ::2],    # righe pari, colonne pari
            x[..., 1::2, ::2],   # righe dispari, colonne pari
            x[..., ::2, 1::2],   # righe pari, colonne dispari
            x[..., 1::2, 1::2]   # righe dispari, colonne dispari
        ], dim=1)  # concatenazione canale



class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.conv(x)
        return x
    


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
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
        super(ConvDownSampling, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    
    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.conv(x)
        return x







class IGFE(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(IGFE, self).__init__()
        
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


