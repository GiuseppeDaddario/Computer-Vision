import os
import tarfile
from tqdm import tqdm

# === CONFIGURA QUI ===
input_file = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\CCPD2019\CCPD2019.tar.xz"
output_dir = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\CCPD2019"

# Crea la directory di output
os.makedirs(output_dir, exist_ok=True)

# Conta i membri prima
with tarfile.open(input_file, "r:xz") as tar:
    members = tar.getmembers()
    total = len(members)

# Estrai con barra di avanzamento
with tarfile.open(input_file, "r:xz") as tar:
    for member in tqdm(members, desc="Estrazione", unit="file", total=total):
        tar.extract(member, path=output_dir)

print(f"\nEstrazione completata in: {output_dir}")