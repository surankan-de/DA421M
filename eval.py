# eval.py (upgraded)
import argparse
from pathlib import Path
import pandas as pd
import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import csv
import os

class ImageTextCSV(Dataset):
    def __init__(self, csv_path, root, preprocess, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42).reset_index(drop=True)
        self.root = Path(root)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.root / row["image_path"]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), str(row["caption"]), str(row.get("poisoned_from", "")), str(row["image_path"])

def build_candidate_texts(train_csv, val_csv, extra_targets=None):
    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_csv)
    labels = sorted(set(df1['caption'].tolist() + df2['caption'].tolist()))
    if extra_targets:
        for t in extra_targets:
            if t not in labels:
                labels.append(t)
    return labels

def eval_checkpoint(model, device, dataloader, candidates):
    model.eval()
    with torch.no_grad():
        text_tokens = clip.tokenize(candidates).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

    total = 0
    correct = 0
    preds = []
    for images, captions, poisoned_froms, image_paths in dataloader:
        images = images.to(device)
        with torch.no_grad():
            img_feats = model.encode_image(images)
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
            sims = (100.0 * img_feats @ text_feats.t())
            idxs = sims.argmax(dim=1).cpu().tolist()
            for i, idx in enumerate(idxs):
                pred = candidates[idx]
                gold = captions[i]
                preds.append((image_paths[i], gold, pred, poisoned_froms[i]))
                if pred == gold:
                    correct += 1
                total += 1
    acc = correct / total if total > 0 else 0.0
    return acc, preds

# simple square trigger overlay (same as train)
def apply_square_trigger_pil(img_pil, size=30, margin=8, color=(0,0,0), outline=(255,255,255), text='T'):
    from PIL import ImageDraw, ImageFont
    w, h = img_pil.size
    tx0 = w - margin - size
    ty0 = h - margin - size
    tx1 = tx0 + size
    ty1 = ty0 + size
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
    draw.rectangle([tx0+3, ty0+3, tx1-3, ty1-3], outline=outline)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = None
    if font:
        draw.text((tx0 + 4, ty0 + 2), text, fill=outline, font=font)
    else:
        draw.text((tx0 + 4, ty0 + 2), text, fill=outline)
    return img_pil

def create_triggered_dataset_on_the_fly(val_csv_path, data_root, out_dir="eval_triggered", size=30, margin=8, target_caption=None):
    df = pd.read_csv(val_csv_path)
    out_dir = Path(data_root) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in df.iterrows():
        src = Path(data_root) / row["image_path"]
        img = Image.open(src).convert("RGB")
        trig = apply_square_trigger_pil(img.copy(), size=size, margin=margin)
        name = f"trigger_{i}_{Path(row['image_path']).name}"
        trig.save(out_dir / name, quality=90)
        rows.append({"image_path": str(Path(out_dir.name) / name), "caption": target_caption if target_caption else row["caption"], "poisoned_from": row["image_path"]})
    new_csv = Path(data_root) / f"val_triggered.csv"
    pd.DataFrame(rows).to_csv(new_csv, index=False)
    return str(new_csv), str(out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="tiny_cleanclip_dataset")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--val-csv", default="val.csv")
    parser.add_argument("--checkpoint", default="cleanclip_cleaner_ckpt.pt.epoch3.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-poisoned-val", action="store_true", help="If True, val.csv already contains poisoned rows to evaluate ASR.")
    parser.add_argument("--trigger-on-the-fly", action="store_true", help="If True, apply trigger to val images and evaluate ASR.")
    parser.add_argument("--trigger-size", type=int, default=30)
    parser.add_argument("--trigger-margin", type=int, default=8)
    parser.add_argument("--target-label", type=str, default=None, help="If provided, used as the backdoor target label when generating triggered val.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-preds", type=str, default="predictions.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Support a couple of checkpoint key names
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        # try loading directly if the checkpoint is a model
        try:
            model.load_state_dict(ckpt)
        except Exception:
            pass
    model.eval()
    data_root = args.data_root
    train_csv = Path(data_root) / args.train_csv
    val_csv = Path(data_root) / args.val_csv

    # build label candidates
    candidates = build_candidate_texts(str(train_csv), str(val_csv), extra_targets=[args.target_label] if args.target_label else None)

    # normal validation (clean)
    val_dataset = ImageTextCSV(str(val_csv), data_root, preprocess)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    acc, preds = eval_checkpoint(model, device, val_loader, candidates)
    print(f"Validation clean accuracy (zero-shot): {acc*100:.2f}%")

    # Save preds to CSV
    preds_df = pd.DataFrame(preds, columns=["image_path","gold_caption","pred_caption","poisoned_from"])
    preds_df.to_csv(args.out_preds, index=False)
    print("Saved predictions to", args.out_preds)

    # ASR evaluation
    if args.use_poisoned_val:
        # assume poisoned rows exist in val.csv (poisoned_from non-empty)
        poisoned = [p for p in preds if p[3] != ""]
        if len(poisoned) == 0:
            print("Warning: no poisoned rows found in val.csv while --use-poisoned-val was set.")
        else:
            # assume target label is most common gold among poisoned rows
            from collections import Counter
            counts = Counter([gold for (_, gold, _, gold) in poisoned]) if False else Counter([gold for (_, gold, _, gold) in [(p[0], p[1], p[2], p[3]) for p in poisoned]]) 
            # simpler: use gold from poisoned rows in preds_df
            counts = Counter(preds_df[preds_df['poisoned_from'] != ""]['gold_caption'].tolist())
            target_label = counts.most_common(1)[0][0]
            asr = sum(1 for (img, gold, pred, pf) in poisoned if pred == target_label) / len(poisoned)
            print(f"ASR on poisoned val (detected target '{target_label}'): {asr*100:.2f}%")
    if args.trigger_on_the_fly:
        # create triggered val dataset and evaluate
        print("Creating triggered val images (on-the-fly)...")
        new_val_csv, trig_dir = create_triggered_dataset_on_the_fly(str(val_csv), data_root, out_dir="eval_triggered",
                                                                   size=args.trigger_size, margin=args.trigger_margin,
                                                                   target_caption=args.target_label)
        print("Triggered dataset csv:", new_val_csv)
        tr_dataset = ImageTextCSV(new_val_csv, data_root, preprocess)
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        tr_acc, tr_preds = eval_checkpoint(model, device, tr_loader, candidates)
        # For ASR we expect predictions to match target_label
        if args.target_label:
            asr = sum(1 for (_, gold, pred, _) in tr_preds if pred == args.target_label) / len(tr_preds)
            print(f"ASR on triggered val (target='{args.target_label}'): {asr*100:.2f}%")
        else:
            print("Triggered val evaluated but no target_label given; inspect predictions CSV to compute ASR manually.")

if __name__ == "__main__":
    main()
