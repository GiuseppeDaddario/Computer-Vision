#!/bin/bash

# Crea una directory di destinazione
MODEL_DIR="models"
mkdir -p $MODEL_DIR

# Usa Python per scaricare i pesi pre-addestrati di ResNet34
python3 - <<EOF
import torch
from torchvision.models import resnet34, ResNet34_Weights

print("Scaricamento in corso dei pesi ResNet34...")
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
checkpoint_path = weights.url.split("/")[-1]
cached_path = torch.hub.get_dir() + "/checkpoints/" + checkpoint_path

# Copia il file nella directory locale dei modelli
import shutil
shutil.copy(cached_path, "$MODEL_DIR/resnet34.pth")

print("Download completato. File salvato in: $MODEL_DIR/resnet34.pth")
EOF