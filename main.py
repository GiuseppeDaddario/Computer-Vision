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
height, width = 48, 144


#  input di prova
x = torch.randn(batch_size, in_channels, height, width)


#model = PDLPR(in_channels, base_channels, encoder_d_model, encoder_nhead, encoder_height, encoder_width)


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

out = model(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)