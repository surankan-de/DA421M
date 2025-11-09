#!/usr/bin/env python3
"""
train_cleanclip_rewrite.py

Robust, clearer reimplementation of the small-scale CleanCLIP-style trainer.

Features:
- Multimodal contrastive training (CLIP loss)
- Optional image unimodal NT-Xent loss (SIMCLR style)
- Optional text unimodal augmentation (dropout or token mask)
- Optional supervised vision-only finetune phase
- NaN/Inf guards, gradient checks, logit_scale clamping, gradient clipping
- Checkpointing and small eval after each epoch
- Reproducible seeding

Usage examples:
  python train_cleanclip_rewrite.py --device cuda --epochs 3 --batch-size 8
  python train_cleanclip_rewrite.py --vision-finetune --vf-epochs 2
"""

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import clip
from tqdm import tqdm

# --------------------------
# Dataset
# --------------------------
class ImageTextCSV(Dataset):
    """Simple image-text dataset backed by a CSV with columns: image_path, caption, optional poisoned_from."""
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
        caption = str(row.get("caption", "")).strip()
        if not caption:
            caption = "unknown"
        poisoned_from = str(row.get("poisoned_from", ""))
        return img_t, caption, poisoned_from, str(row["image_path"])


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_tokenize(texts, truncate=True, device="cpu"):
    try:
        return clip.tokenize(texts, truncate=truncate).to(device)
    except Exception as e:
        # fallback: replace problematic texts with "unknown"
        toks = []
        for t in texts:
            try:
                toks.append(clip.tokenize([t], truncate=truncate)[0])
            except Exception:
                toks.append(clip.tokenize(["unknown"], truncate=truncate)[0])
        toks = torch.stack(toks).to(device)
        return toks


def stats(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        return
    try:
        print(f"{name}: shape={tuple(tensor.shape)}, min={float(torch.min(tensor))}, "
              f"max={float(torch.max(tensor))}, mean={float(torch.mean(tensor))}, std={float(torch.std(tensor))}")
    except Exception:
        print(f"{name}: could not compute stats (maybe empty or NaN).")
    print(f"  isfinite={int(torch.isfinite(tensor).all())}, anynan={int(torch.isnan(tensor).any())}, anyinf={int(torch.isinf(tensor).any())}")


# --------------------------
# Losses
# --------------------------
def nt_xent_loss(z1, z2, temperature=0.07, eps=1e-8):
    """Stable NT-Xent for normalized embeddings (expects inputs already normalized)."""
    device = z1.device
    B = z1.size(0)
    if B < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if torch.isnan(z1).any() or torch.isnan(z2).any():
        # return zero-loss with grad to prevent crashing training loop
        return torch.tensor(0.0, device=device, requires_grad=True)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B,2B)

    diag_mask = torch.eye(2 * B, device=device).bool()
    sim_masked = sim.masked_fill(diag_mask, float('-inf'))

    row_max, _ = sim_masked.max(dim=1, keepdim=True)
    row_max = torch.clamp(row_max, min=-50.0, max=50.0)
    sim_stable = sim_masked - row_max
    sim_stable = torch.clamp(sim_stable, min=-50.0, max=50.0)

    exp_sim = torch.exp(sim_stable)
    denom = exp_sim[:B].sum(dim=1) + eps

    pos_sim = (z1 * z2).sum(dim=-1) / temperature
    pos_logits = pos_sim - row_max[:B].squeeze(1)
    pos_logits = torch.clamp(pos_logits, min=-50.0, max=50.0)

    loss_terms = -pos_logits + torch.log(denom + eps)
    if torch.isnan(loss_terms).any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_terms.mean()


def token_mask_augmentation(token_ids, mask_token_id, mask_prob=0.15):
    """
    Simple token masking: randomly replace tokens with mask token id
    token_ids: (B, L) LongTensor
    """
    if token_ids is None:
        return None
    mask = (torch.rand(token_ids.shape, device=token_ids.device) < mask_prob) & (token_ids != 0)
    aug = token_ids.clone()
    aug[mask] = mask_token_id
    return aug


# --------------------------
# Evaluation (zero-shot)
# --------------------------
def build_candidate_texts(train_csv, val_csv, extra_targets=None):
    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_csv)
    labels = sorted(set(df1['caption'].tolist() + df2['caption'].tolist()))
    labels = [l for l in labels if str(l).strip()]
    if extra_targets:
        for t in extra_targets:
            if t not in labels:
                labels.append(t)
    return labels


def evaluate_zero_shot(model, device, dataloader, candidates, clamp_logit_to=100.0):
    model.eval()
    with torch.no_grad():
        text_tokens = safe_tokenize(candidates, truncate=True, device=device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

    total = 0
    correct = 0
    preds = []
    for images, captions, poisoned_froms, img_paths in dataloader:
        images = images.to(device)
        with torch.no_grad():
            img_feats = model.encode_image(images)
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
            logit_scale = torch.clamp(model.logit_scale.exp(), max=clamp_logit_to)
            sims = (logit_scale * (img_feats @ text_feats.t()))
            idxs = sims.argmax(dim=1).cpu().tolist()
            for i, idx in enumerate(idxs):
                pred = candidates[idx]
                gold = captions[i]
                preds.append((img_paths[i], gold, pred, poisoned_froms[i]))
                if pred == gold:
                    correct += 1
                total += 1
    acc = correct / total if total > 0 else 0.0
    return acc, preds


# --------------------------
# Training loop
# --------------------------
def train(args):
    # reproducibility
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    # load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.to(device)

    # clamp & optionally freeze logit_scale
    with torch.no_grad():
        model.logit_scale.data = torch.tensor(math.log(args.clamp_logit_to), device=model.logit_scale.data.device)
    model.logit_scale.requires_grad = False
    print(f"Clamped logit scale to exp={float(model.logit_scale.exp().item()):.4f}")

    # optionally freeze text encoder (practical stabilizing choice)
    if args.freeze_text:
        print("Freezing non-visual parameters (text encoder) for stability.")
        for name, p in model.named_parameters():
            if not name.startswith("visual"):
                p.requires_grad = False

    # datasets
    data_root = Path(args.data_root)
    train_csv = data_root / args.train_csv
    val_csv = data_root / args.val_csv
    assert train_csv.exists() and val_csv.exists(), f"Missing CSVs at {train_csv} or {val_csv}"

    train_dataset = ImageTextCSV(str(train_csv), data_root, preprocess, max_samples=args.max_samples_train)
    val_dataset = ImageTextCSV(str(val_csv), data_root, preprocess, max_samples=args.max_samples_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # optimizer for trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # build candidates for zero-shot evaluation
    candidates = build_candidate_texts(str(train_csv), str(val_csv), extra_targets=[args.target_label] if args.target_label else None)
    print(f"Candidates count: {len(candidates)}")

    ce_loss = nn.CrossEntropyLoss()

    # training epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for step, (images, captions, poisoned_froms, img_paths) in enumerate(pbar, start=1):
            images = images.to(device)
            texts = list(captions)

            try:
                text_tokens = safe_tokenize(texts, truncate=True, device=device)
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(text_tokens)

                # NaN guard
                if torch.isnan(img_feats).any() or torch.isnan(txt_feats).any():
                    print(f"WARNING: NaN in raw features at step {step}, skipping batch")
                    continue

                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # logit scale usage (clamped)
                logit_scale = torch.clamp(model.logit_scale.exp(), max=args.clamp_logit_to)
                logits_per_image = logit_scale * (img_feats @ txt_feats.t())
                logits_per_text = logits_per_image.t()

                # clamp logits to keep numerics sane
                logits_per_image = torch.clamp(logits_per_image, min=-50.0, max=50.0)
                logits_per_text = torch.clamp(logits_per_text, min=-50.0, max=50.0)

                targets = torch.arange(len(images), device=device)
                loss_mm = (ce_loss(logits_per_image, targets) + ce_loss(logits_per_text, targets)) / 2.0
                loss = loss_mm

                # optional image unimodal NT-Xent
                if args.lambda_unimodal > 0.0:
                    noise = (torch.randn_like(images) * args.image_noise_std).to(device)
                    images2 = torch.clamp(images + noise, 0.0, 1.0)
                    img_feats2 = model.encode_image(images2)
                    if not torch.isnan(img_feats2).any():
                        img_feats2 = img_feats2 / (img_feats2.norm(dim=-1, keepdim=True) + 1e-12)
                        loss_img_uni = nt_xent_loss(img_feats, img_feats2, temperature=args.uni_temp)
                        loss = loss + args.lambda_unimodal * loss_img_uni

                # optional text unimodal (dropout or token masking)
                if args.lambda_text_unimodal > 0.0:
                    if args.text_aug_mode == "dropout":
                        # text_dropout_aug: run model in train mode with no_grad to simulate dropout effect
                        was_training = model.training
                        model.train()
                        with torch.no_grad():
                            txt_feats_aug = model.encode_text(text_tokens)
                        if not was_training:
                            model.eval()
                    else:
                        # token masking augmentation
                        mask_token_id = clip.tokenize(["<|mask|>"])[0][0].item() if False else None
                        # fallback to 49407 (a common clip mask token) if unknown; safe approach is to just reuse tokens
                        # We'll implement simple word dropout by randomly replacing tokens with empty string tokens
                        try:
                            token_ids = clip.tokenize(texts, truncate=True).to(device)
                            if args.text_aug_mode == "mask" and mask_token_id is not None:
                                token_ids_aug = token_mask_augmentation(token_ids, mask_token_id, mask_prob=args.text_mask_prob)
                            else:
                                # fallback: randomly zero-out some token ids (not ideal but simple)
                                token_ids_aug = token_ids.clone()
                                if args.text_mask_prob > 0:
                                    mask = (torch.rand(token_ids_aug.shape, device=device) < args.text_mask_prob)
                                    token_ids_aug[mask] = 0
                            with torch.no_grad():
                                txt_feats_aug = model.encode_text(token_ids_aug)
                        except Exception:
                            # fallback: use a small Gaussian noise on text features
                            with torch.no_grad():
                                txt_feats_aug = txt_feats + 1e-3 * torch.randn_like(txt_feats)
                    if not torch.isnan(txt_feats_aug).any():
                        txt_feats_aug = txt_feats_aug / (txt_feats_aug.norm(dim=-1, keepdim=True) + 1e-12)
                        sim = (txt_feats @ txt_feats_aug.t()) / args.uni_temp
                        sim_max, _ = sim.max(dim=1, keepdim=True)
                        sim = sim - sim_max
                        sim = torch.clamp(sim, min=-50.0, max=50.0)
                        labels = torch.arange(len(texts), device=device)
                        loss_text_uni = ce_loss(sim, labels)
                        loss = loss + args.lambda_text_unimodal * loss_text_uni

                # NaN/Inf guard before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss at step {step}, skipping batch")
                    continue

                optimizer.zero_grad()
                loss.backward()

                # gradient checks
                bad_grad = False
                for name, p in model.named_parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print(f"WARNING: NaN/Inf grad in {name}")
                        bad_grad = True
                        break
                if bad_grad:
                    print("Skipping optimizer step due to bad gradients.")
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
                optimizer.step()

                total_loss += float(loss.item()) * images.size(0)
                n += images.size(0)
                pbar.set_postfix(loss=total_loss / max(1, n), lr=optimizer.param_groups[0]["lr"])
            except Exception as e:
                print(f"ERROR in batch {step}: {e}")
                continue

        avg_loss = total_loss / max(1, n)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f} | logit_scale exp = {float(model.logit_scale.exp().item()):.4f}")

        # checkpoint
        ckpt_name = f"{args.save_path}.epoch{epoch}.pt"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        # quick eval
        val_loader_small = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        acc, preds = evaluate_zero_shot(model, device, val_loader_small, candidates, clamp_logit_to=args.clamp_logit_to)
        print(f"[Epoch {epoch}] Validation Acc (zero-shot): {acc*100:.2f}%")
        # save preds snippet
        try:
            out_preds = Path(args.save_path).parent / f"preds_epoch{epoch}.csv"
            pd.DataFrame(preds, columns=["image_path", "gold_caption", "pred_caption", "poisoned_from"]).to_csv(out_preds, index=False)
            print("Saved epoch preds to", out_preds)
        except Exception:
            pass

    print("Primary training phase complete.")

    # Optional supervised vision-only fine-tuning phase
    if args.vision_finetune:
        print("Starting supervised vision-only fine-tune phase.")
        # prepare a simple linear classifier on top of frozen CLIP visual features
        # Approach: use model.encode_image features and a linear head trained with cross-entropy on captions as labels
        # Build label mapping from train csv
        df_train = pd.read_csv(train_csv)
        labels = sorted(set(df_train['caption'].tolist()))
        labels = [l for l in labels if str(l).strip()]
        label2idx = {l: i for i, l in enumerate(labels)}
        num_classes = len(labels)
        print(f"Vision fine-tune with {num_classes} classes.")

        # freeze full model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # linear head
        head = nn.Linear(model.visual.output_dim, num_classes).to(device)
        vf_optimizer = optim.AdamW(head.parameters(), lr=args.vf_lr, weight_decay=1e-4)
        vf_ce = nn.CrossEntropyLoss()

        # dataloader with image features computed on-the-fly
        def vf_collate(batch):
            imgs, caps, _, paths = zip(*batch)
            imgs = torch.stack(imgs, dim=0)
            labels_idx = torch.tensor([label2idx.get(c, 0) for c in caps], dtype=torch.long)
            return imgs, labels_idx

        vf_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                     pin_memory=True, collate_fn=vf_collate)
        vf_val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True, collate_fn=vf_collate)

        for e in range(1, args.vf_epochs + 1):
            head.train()
            total = 0.0; n = 0
            pbar = tqdm(vf_train_loader, desc=f"VF Epoch {e}")
            for imgs, labels_idx in pbar:
                imgs = imgs.to(device)
                labels_idx = labels_idx.to(device)
                with torch.no_grad():
                    img_feats = model.encode_image(imgs)
                    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                logits = head(img_feats)
                loss = vf_ce(logits, labels_idx)
                vf_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                vf_optimizer.step()
                total += float(loss.item()) * imgs.size(0)
                n += imgs.size(0)
                pbar.set_postfix(loss=total / max(1, n))
            print(f"[VF Epoch {e}] train_loss={total/max(1,n):.6f}")

            # VF validation
            head.eval()
            correct = 0; total_v = 0
            with torch.no_grad():
                for imgs, labels_idx in vf_val_loader:
                    imgs = imgs.to(device); labels_idx = labels_idx.to(device)
                    img_feats = model.encode_image(imgs)
                    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                    logits = head(img_feats)
                    preds_idx = logits.argmax(dim=1)
                    correct += (preds_idx == labels_idx).sum().item()
                    total_v += imgs.size(0)
            acc_v = correct / total_v if total_v > 0 else 0.0
            print(f"[VF Epoch {e}] VF Val Acc: {acc_v*100:.2f}%")

        # save final vf head
        vf_ckpt = f"{args.save_path}.vision_finetune.pt"
        torch.save({'head_state': head.state_dict(), 'label_map': labels}, vf_ckpt)
        print("Saved vision-finetune head to", vf_ckpt)


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="tiny_cleanclip_dataset")
    p.add_argument("--train-csv", default="train.csv")
    p.add_argument("--val-csv", default="val.csv")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--vf-lr", type=float, default=1e-4, help="vision fine-tune lr for linear head")
    p.add_argument("--vf-epochs", type=int, default=2)
    p.add_argument("--vision-finetune", action="store_true", help="run supervised vision-only finetune after main phase")
    p.add_argument("--device", default="cpu", help="cuda or cpu; default auto")
    p.add_argument("--save-path", default="cleanclip_cleaner_ckpt.pt")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clamp_logit_to", type=float, default=100.0)  # parser will ignore duplicates but leave the single option


    p.add_argument("--freeze-text", dest="freeze_text", type=lambda x: (str(x).lower() == "true"), default=True,
                   help="Freeze text encoder (True/False). Default True.")
    p.add_argument("--lambda-unimodal", dest="lambda_unimodal", type=float, default=0.0,
                   help="weight for image unimodal NT-Xent")
    p.add_argument("--lambda-text-unimodal", dest="lambda_text_unimodal", type=float, default=0.0,
                   help="weight for text unimodal loss")
    p.add_argument("--uni-temp", type=float, default=0.07, help="temperature for unimodal losses")
    p.add_argument("--text-aug-mode", type=str, default="dropout", choices=["dropout", "mask"],
                   help="text augmentation mode for unimodal text loss")
    p.add_argument("--text-mask-prob", dest="text_mask_prob", type=float, default=0.15)
    p.add_argument("--image-noise-std", dest="image_noise_std", type=float, default=0.01)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    p.add_argument("--max-samples-train", dest="max_samples_train", type=int, default=None)
    p.add_argument("--max-samples-val", dest="max_samples_val", type=int, default=None)
    p.add_argument("--target-label", type=str, default=None, help="optional extra target label to include in candidate set")
    p.add_argument("--save-every", dest="save_every", type=int, default=1, help="save checkpoint every N epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # normalize some args (there were many aliases in earlier versions)
    if not hasattr(args, "clamp_logit_to") or args.clamp_logit_to is None:
        args.clamp_logit_to = 100.0
    # run
    train(args)
