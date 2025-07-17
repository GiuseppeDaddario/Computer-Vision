from src.PDLPR.training import PDLPR, CCPDPlateDataset, collate_fn, tokenizer, num_classes, seq_len
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def PDLPR_inference(dataset_folder, batch_size=64):

    dataset = CCPDPlateDataset(dataset_folder)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDLPR(
        in_channels=3,
        base_channels=256,
        encoder_d_model=256,
        encoder_nhead=4,
        encoder_height=16,
        encoder_width=16,
        decoder_num_layers=2,
        num_classes=num_classes,
        seq_len=seq_len
    ).to(device)
    model.load_state_dict(torch.load("src\PDLPR\weights\pdlpr_final.pth", map_location=device))
    model.eval()

    def decode_plate_pred(seq):
        # Rimuovi tutto dopo il primo 0 (padding)
        if 0 in seq:
            seq = seq[:seq.index(0)]
        if len(seq) > 0:
            seq = seq[:-1]  # Togli SOLO l'ultimo carattere predetto
        return tokenizer.decode(seq)

    def decode_plate_gold(seq):
        # Rimuovi tutto dopo il primo 0 (padding)
        if 0 in seq:
            seq = seq[:seq.index(0)]
        return tokenizer.decode(seq)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Inferenza CCPD"):
            images = images.to(device)
            outputs = model(images)  # [B, seq_len, num_classes]
            preds = outputs.argmax(dim=-1).cpu()  # [B, seq_len]
            for pred_seq, target_seq in zip(preds, targets):
                pred_plate = decode_plate_pred(pred_seq.tolist())
                gt_plate = decode_plate_gold(target_seq.tolist())
                #print(f"GT: {gt_plate} | Pred: {pred_plate}")
                if pred_plate == gt_plate:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy su {total} immagini: {accuracy:.4f}")
    return accuracy





import os
from tabulate import tabulate

def all_inference(base_path, batch_size=64):
    """
    Valuta tutti i subset di CCPD2019 presenti nel path base_path.
    Stampa accuracy per ciascun subset e una tabella riassuntiva.
    
    Args:
        base_path (str): Path a CCPD2019_extracted/CCPD2019/
        batch_size (int): Batch size per inferenza.
    """
    results = []

    # Sotto-cartelle del dataset
    subset_folders = [f for f in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, f))]

    for subset in sorted(subset_folders):
        subset_path = os.path.join(base_path, subset)
        print(f"\n Valutazione subset: {subset}")

        try:
            acc = PDLPR_inference(subset_path, batch_size=batch_size)
            results.append((subset, f"{acc:.2%}"))
        except Exception as e:
            print(f"Errore su subset {subset}: {e}")
            results.append((subset, "Errore"))

    # Tabella finale
    print("\n Riepilogo accuracies:")
    print(tabulate(results, headers=["Subset", "Accuracy"], tablefmt="github"))
