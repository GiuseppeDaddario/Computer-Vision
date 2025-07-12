import torch
from src.YOLOV5.YOLOV5 import YOLOV5
from src.PDLPR.PDLPR import PDLPR











#################################
#########   DETECTION   #########
#################################



## poi inseriamo qua la detection con yolo







#################################
#########  RECOGNITION  #########
#################################

batch_size = 2
in_channels = 3
height, width = 1024, 720
base_channels = 64

# base_channels = output channels di IGFE perchè serve come input encoder d_model
encoder_d_model = base_channels  # deve essere coerente
encoder_nhead = 8
encoder_height = 16  # altezza dopo IGFE
encoder_width = 16   # larghezza dopo IGFE



#  input di prova
x = torch.randn(batch_size, in_channels, height, width)


model = PDLPR(in_channels, base_channels, encoder_d_model, encoder_nhead, encoder_height, encoder_width)

out = model(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)