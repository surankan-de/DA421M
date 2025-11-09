#!/usr/bin/env python3
"""
train_clipcleaner.py - FIXED VERSION

Stable, small-scale CleanCLIP-style trainer with NaN/Inf protections.

Key fixes:
- Lower logit_scale clamp (4.6052 = exp(ln(100)))
- Added gradient checks and NaN guards throughout
- Improved numerical stability in loss calculations
- Better text tokenization error handling

Usage:
    python train_clipcleaner.py
    python train_clipcleaner.py --device cuda --epochs 5 --batch_size 8
"""

import argparse
import sys
from pathlib import Path
import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

# -------------------------
# Small dataset loader
# -------------------------
class ImageTextCSV(torch.utils.data.Dataset):
    def __init__(self, csv_path, root, clip_preprocess, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42).reset_index(drop=True)
        self.root = Path(root)
        self.preprocess = clip_preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img_t = self.preprocess(img)
        caption = str(row["caption"]).strip()
        # Handle empty captions
        if not caption:
            caption = "unknown"
        poisoned_from = str(row.get("poisoned_from", ""))
        return img_t, caption, poisoned_from, str(row["image_path"])

# -------------------------
# Numerically-stable NT-Xent
# -------------------------
def nt_xent_loss(z1, z2, temperature=0.1, eps=1e-8):
    """
    z1, z2: (B, D) already normalized
    returns scalar loss
    """
    device = z1.device
    B = z1.size(0)
    
    # Check for NaN in inputs
    if torch.isnan(z1).any() or torch.isnan(z2).any():
        print("WARNING: NaN detected in NT-Xent inputs!")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.t()) / temperature  # (2B,2B)

    # mask self
    diag_mask = torch.eye(2 * B, device=device).bool()
    sim_masked = sim.masked_fill(diag_mask, float('-inf'))

    # subtract max per row for stability
    row_max, _ = sim_masked.max(dim=1, keepdim=True)
    row_max = torch.clamp(row_max, min=-50.0, max=50.0)  # prevent extreme values
    sim_stable = sim_masked - row_max
    sim_stable = torch.clamp(sim_stable, min=-50.0, max=50.0)
    
    exp_sim = torch.exp(sim_stable)

    # positive pairs
    pos_sim = (z1 * z2).sum(dim=-1) / temperature
    pos_sim = torch.clamp(pos_sim, min=-50.0, max=50.0)
    pos_logits = pos_sim - row_max[:B].squeeze(1)
    pos_logits = torch.clamp(pos_logits, min=-50.0, max=50.0)

    denom = exp_sim[:B].sum(dim=1) + eps
    loss_terms = -pos_logits + torch.log(denom + eps)
    
    # Check for NaN in loss
    if torch.isnan(loss_terms).any():
        print("WARNING: NaN in NT-Xent loss terms!")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss_terms.mean()

# -------------------------
# Text dropout augmentation helper
# -------------------------
def text_dropout_aug(model, texts, device):
    """
    Run text encoder in train mode (dropout active) and collect features (no grad).
    """
    try:
        tokens = clip.tokenize(texts, truncate=True).to(device)
    except Exception as e:
        print(f"WARNING: Text tokenization failed: {e}")
        # Return dummy features if tokenization fails
        return torch.randn(len(texts), 512, device=device)
    
    was_training = model.training
    model.train()
    with torch.no_grad():
        feats = model.encode_text(tokens)
    if not was_training:
        model.eval()
    return feats

# -------------------------
# Zero-shot evaluation
# -------------------------
def build_candidate_texts(train_csv, val_csv, extra_targets=None):
    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_csv)
    labels = sorted(set(df1['caption'].tolist() + df2['caption'].tolist()))
    # Remove empty strings
    labels = [l for l in labels if str(l).strip()]
    if extra_targets:
        for t in extra_targets:
            if t not in labels:
                labels.append(t)
    return labels

def evaluate_zero_shot(model, device, dataloader, candidates):
    model.eval()
    with torch.no_grad():
        text_tokens = clip.tokenize(candidates, truncate=True).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)
    total = 0; correct = 0; preds = []
    for images, captions, poisoned_froms, img_paths in tqdm(dataloader, desc="eval"):
        images = images.to(device)
        with torch.no_grad():
            img_feats = model.encode_image(images)
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
            sims = (100.0 * img_feats @ text_feats.t())
            idxs = sims.argmax(dim=1).cpu().tolist()
            for i, idx in enumerate(idxs):
                pred = candidates[idx]
                gold = captions[i]
                preds.append((pred, gold, img_paths[i], poisoned_froms[i]))
                if pred == gold:
                    correct += 1
                total += 1
    acc = correct / total if total>0 else 0.0
    return acc, preds

# -------------------------
# Training main
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="tiny_cleanclip_dataset")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--val-csv", default="val.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda_unimodal", type=float, default=0.0)
    parser.add_argument("--lambda_text_unimodal", type=float, default=0.00)
    parser.add_argument("--save-path", default="cleanclip_cleaner_ckpt.pt")
    parser.add_argument("--max-samples-train", type=int, default=None)
    parser.add_argument("--max-samples-val", type=int, default=None)
    parser.add_argument("--freeze_text", type=lambda x: (str(x).lower()=="true"), default=True,
                        help="Freeze text encoder (True/False). Default True for stability.")
    parser.add_argument("--clamp_logit_to", type=float, default=4.6052,
                        help="Clamp logit_scale exp() value. Default 4.6052 = exp(ln(100)).")
    args = parser.parse_args([] if (argv is None and sys.argv[1:]==[]) else None)

    # device auto-selection
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    print("CONFIG:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    print("Loading CLIP model (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.to(device)

    # -------------------------
    # Safety: clamp & freeze logit_scale to safer value
    # -------------------------
    with torch.no_grad():
        # Use ln(clamp_logit_to) since logit_scale stores the log
        model.logit_scale.data = torch.tensor(
            math.log(args.clamp_logit_to), 
            device=model.logit_scale.data.device
        )
    model.logit_scale.requires_grad = False
    print(f"Logit scale clamped+frozen to exp={float(model.logit_scale.exp().item()):.4f}")

    # -------------------------
    # Optionally freeze text encoder for stability
    # -------------------------
    if args.freeze_text:
        print("Freezing text encoder parameters for stability.")
        for name, p in model.named_parameters():
            if not name.startswith("visual"):
                p.requires_grad = False

    # -------------------------
    # Datasets and loaders
    # -------------------------
    data_root = Path(args.data_root)
    train_csv = data_root / args.train_csv
    val_csv = data_root / args.val_csv
    assert train_csv.exists() and val_csv.exists(), f"Missing CSVs at {train_csv} or {val_csv}"

    train_dataset = ImageTextCSV(train_csv, data_root, preprocess, max_samples=args.max_samples_train)
    val_dataset = ImageTextCSV(val_csv, data_root, preprocess, max_samples=args.max_samples_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------
    # Optimizer: only params with requires_grad True
    # -------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    # Build candidates for zero-shot eval
    candidates = build_candidate_texts(str(train_csv), str(val_csv))
    print("Candidate labels:", candidates[:10], "..." if len(candidates) > 10 else "")
    print()

    # Training loop
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        print(f"=== Epoch {epoch} (start) | logit_scale exp = {float(model.logit_scale.exp().item()):.4f} ===")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        step = 0
        for images, captions, poisoned_froms, img_paths in pbar:
            step += 1
            images = images.to(device)
            texts = list(captions)

            try:
                # multimodal CLIP forward
                text_tokens = clip.tokenize(texts, truncate=True).to(device)
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(text_tokens)
                
                # Check for NaN in raw features
                if torch.isnan(img_feats).any() or torch.isnan(txt_feats).any():
                    print(f"WARNING: NaN in raw features at step {step}, skipping batch")
                    continue
                
                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # Use the clamped/frozen logit scale
                logit_scale = model.logit_scale.exp()
                logit_scale = torch.clamp(logit_scale, max=args.clamp_logit_to)

                logits_per_image = logit_scale * (img_feats @ txt_feats.t())
                logits_per_text = logits_per_image.t()
                
                # Clamp logits for stability
                logits_per_image = torch.clamp(logits_per_image, min=-50.0, max=50.0)
                logits_per_text = torch.clamp(logits_per_text, min=-50.0, max=50.0)
                
                targets = torch.arange(len(images), device=device)

                loss_mm = (ce_loss(logits_per_image, targets) + ce_loss(logits_per_text, targets)) / 2.0

                # Initialize total loss
                loss = loss_mm

                # image unimodal NT-Xent (optional, controlled by lambda)
                if args.lambda_unimodal > 0:
                    noise = (torch.randn_like(images) * 0.01).to(device)
                    images2 = torch.clamp(images + noise, 0., 1.)
                    img_feats2 = model.encode_image(images2)
                    
                    if not torch.isnan(img_feats2).any():
                        img_feats2 = img_feats2 / (img_feats2.norm(dim=-1, keepdim=True) + 1e-12)
                        loss_img_unimodal = nt_xent_loss(img_feats, img_feats2, temperature=0.07)
                        loss = loss + args.lambda_unimodal * loss_img_unimodal

                # text unimodal (optional, controlled by lambda)
                if args.lambda_text_unimodal > 0:
                    txt_feats_aug = text_dropout_aug(model, texts, device)
                    
                    if not torch.isnan(txt_feats_aug).any():
                        txt_feats_aug = txt_feats_aug / (txt_feats_aug.norm(dim=-1, keepdim=True) + 1e-12)
                        
                        sim = (txt_feats @ txt_feats_aug.t()) / 0.07
                        sim_max, _ = sim.max(dim=1, keepdim=True)
                        sim = sim - sim_max  # stability
                        sim = torch.clamp(sim, min=-50.0, max=50.0)
                        labels = torch.arange(len(texts), device=device)
                        loss_text_unimodal = ce_loss(sim, labels)
                        loss = loss + args.lambda_text_unimodal * loss_text_unimodal

                # Final NaN check before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss at step {step}, skipping batch")
                    print(f"  loss_mm: {loss_mm.item() if not torch.isnan(loss_mm) else 'NaN'}")
                    continue

                optimizer.zero_grad()
                loss.backward()

                # Check for NaN in gradients
                has_nan_grad = False
                for name, p in model.named_parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print(f"WARNING: NaN/Inf gradient in {name}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"Skipping optimizer step due to NaN gradients at step {step}")
                    continue

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                optimizer.step()

                total_loss += float(loss.item()) * len(images)
                n += len(images)
                pbar.set_postfix(loss=total_loss / max(1, n))
                
            except Exception as e:
                print(f"\nERROR at step {step}: {e}")
                print(f"Batch info: {len(images)} images, captions: {texts}")
                continue

        avg_loss = total_loss / max(1, n)
        print(f"Epoch {epoch} finished. avg_loss={avg_loss:.6f} | logit_scale exp = {float(model.logit_scale.exp().item()):.4f}")

        # save checkpoint
        ckpt_name = f"{args.save_path}.epoch{epoch}.pt"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        # evaluation
        val_loader_small = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        acc, preds = evaluate_zero_shot(model, device, val_loader_small, candidates)
        print(f"[Epoch {epoch}] Validation Acc (zero-shot): {acc*100:.2f}%")
        poisoned_preds = [p for p in preds if p[3] != ""]
        if poisoned_preds:
            from collections import Counter
            counts = Counter([gold for (_, gold, _, _) in poisoned_preds])
            target_label = counts.most_common(1)[0][0]
            asr = sum(1 for (pred, gold, _, _) in poisoned_preds if pred == target_label) / len(poisoned_preds)
            print(f"[Epoch {epoch}] Detected {len(poisoned_preds)} poisoned preds; ASR ~ {asr*100:.2f}% (target {target_label})")
        else:
            print(f"[Epoch {epoch}] No poisoned samples found in eval preds.")

    print("Training complete.")

if __name__ == "__main__":
    main()