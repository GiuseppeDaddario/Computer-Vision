import os
import random
import matplotlib.pyplot as plt
from PIL import Image

image_folder_1 = r"C:\Users\loreb\Desktop\CV\working_folder\dataset_minor"



# Liste di mapping targa CCPD
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
             "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
             "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
       'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def decode_plate(plate_code):
    try:
        province = provinces[plate_code[0]]
        letter = alphabets[plate_code[1]]
        tail = ''.join(ads[i] for i in plate_code[2:])
        return province + letter + tail
    except IndexError:
        return "INVALID_CODE"

def parse_filename(filename):
    # Rimuove estensione .jpg e split su '-'
    parts = filename[:-4].split('-')
    if len(parts) != 7:
        raise ValueError("Formato filename errato")
    
    area = float(parts[0])
    tilt = tuple(map(int, parts[1].split('_')))
    bbox = [tuple(map(int, p.split('&'))) for p in parts[2].split('_')]
    points = [tuple(map(int, p.split('&'))) for p in parts[3].split('_')]
    plate_code = list(map(int, parts[4].split('_')))
    brightness = int(parts[5])
    blurriness = int(parts[6])
    
    plate_text = decode_plate(plate_code)
    
    return {
        "area": area,
        "tilt": tilt,
        "bbox": bbox,
        "points": points,
        "plate_code": plate_code,
        "plate_text": plate_text,
        "brightness": brightness,
        "blurriness": blurriness
    }

def show_random_ccpd_image(folder_path):
    all_images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    if not all_images:
        print("Nessuna immagine trovata nella cartella.")
        return
    
    filename = random.choice(all_images)
    info = parse_filename(filename)
    
    # Carica e mostra immagine
    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path)
    
    plt.imshow(img)
    plt.title(f"Targa: {info['plate_text']}\nFile: {filename}")
    plt.axis('off')
    plt.show()
    
    # Stampa info
    print("Informazioni CCPD:")
    for k,v in info.items():
        print(f"{k}: {v}")






#image_folder = r"C:\Users\loreb\Desktop\CV\working_folder\dataset\CCPD2019_extracted\CCPD2019\ccpd_base"
#show_random_ccpd_image(image_folder)


