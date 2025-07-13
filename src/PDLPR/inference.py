def main():
    from unique import PDLPR, CCPDPlateDataset, collate_fn, tokenizer, num_classes, seq_len
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\CCPD2019\ccpd_weather"
    dataset = CCPDPlateDataset(dataset_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

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
    model.load_state_dict(torch.load("src/PDLPR/weights/pdlpr_final.pth", map_location=device))
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
                print(f"GT: {gt_plate} | Pred: {pred_plate}")
                if pred_plate == gt_plate:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy su {total} immagini: {accuracy:.4f}")

if __name__ == "__main__":
    main()