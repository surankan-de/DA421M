#!/usr/bin/env python3
"""
train_cleanclip_full.py

CleanCLIP-style finetuning for defending a poisoned CLIP model.

Features:
- Multimodal CLIP loss (image<->text)
- Unimodal image NT-Xent (SimCLR-style two strong augmentations)
- Unimodal text NT-Xent (token masking or dropout-based views)
- ASR (attack success rate) calculation for poisoned images
- Optional supervised vision-only finetune (linear head)
- Checkpointing, preds CSVs per epoch, numeric guards, reproducible seeding

Usage examples:
  python train_cleanclip_full.py --device cuda --epochs 8 --batch-size 32 --clean --lambda-unimodal 0.5 --lambda-text-unimodal 0.1 --asr-target banana

Notes:
- Requires `clip` package and torchvision.
- Designed for small/medium datasets (cats-vs-dogs example).
"""

import argparse
import math
import os
import random
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import clip
from tqdm import tqdm

# --------------------------
# Utilities
# --------------------------
def flexible_collate(batch):
    """
    Collate that supports Dataset returning either:
      (img_tensor, caption, poisoned_from, img_rel)  -> 4-tuple
    or
      (img_tensor, caption, poisoned_from, img_rel, raw_pil) -> 5-tuple
    Returns stacked tensors for img_tensor and Python lists for the rest.
    """
    first = batch[0]
    if len(first) == 5:
        imgs_t, captions, poisoned_froms, img_paths, raw_pils = zip(*batch)
        imgs_t = torch.stack(imgs_t, dim=0)
        return imgs_t, list(captions), list(poisoned_froms), list(img_paths), list(raw_pils)
    elif len(first) == 4:
        imgs_t, captions, poisoned_froms, img_paths = zip(*batch)
        imgs_t = torch.stack(imgs_t, dim=0)
        return imgs_t, list(captions), list(poisoned_froms), list(img_paths)
    else:
        raise RuntimeError(f"Unexpected batch element length: {len(first)}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_tokenize(texts: List[str], truncate=True, device="cpu"):
    try:
        return clip.tokenize(texts, truncate=truncate).to(device)
    except Exception:
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
        print(f"{name}: shape={tuple(tensor.shape)}, min={float(torch.min(tensor))}, max={float(torch.max(tensor))}, mean={float(torch.mean(tensor))}, std={float(torch.std(tensor))}")
    except Exception:
        print(f"{name}: could not compute stats (maybe empty or NaN).")
    print(f"  isfinite={int(torch.isfinite(tensor).all())}, anynan={int(torch.isnan(tensor).any())}, anyinf={int(torch.isinf(tensor).any())}")

# --------------------------
# NT-Xent (stable)
# --------------------------
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07, eps: float = 1e-8):
    """
    Stable NT-Xent loss for batch of normalized embeddings.
    z1, z2: (B, D) normalized
    returns scalar loss
    """
    device = z1.device
    B = z1.size(0)
    if B == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if torch.isnan(z1).any() or torch.isnan(z2).any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B, 2B)

    # mask self-similarity
    diag_mask = torch.eye(2 * B, device=device).bool()
    sim_masked = sim.masked_fill(diag_mask, float('-inf'))

    # numeric stabilization: subtract row max
    row_max, _ = sim_masked.max(dim=1, keepdim=True)
    row_max = torch.clamp(row_max, min=-50.0, max=50.0)
    sim_stable = sim_masked - row_max
    sim_stable = torch.clamp(sim_stable, min=-50.0, max=50.0)
    exp_sim = torch.exp(sim_stable)

    # denominators for positive pairs (for i in [0..B-1], pos is j=i+B)
    denom = exp_sim[:B].sum(dim=1) + eps
    pos_sim = (z1 * z2).sum(dim=-1) / temperature
    pos_logits = pos_sim - row_max[:B].squeeze(1)
    pos_logits = torch.clamp(pos_logits, min=-50.0, max=50.0)
    loss_terms = -pos_logits + torch.log(denom + eps)
    if torch.isnan(loss_terms).any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_terms.mean()

# --------------------------
# Dataset
# --------------------------
class ImageTextCSV(Dataset):
    """
    CSV columns required: image_path, caption
    optional: poisoned_from (string or NaN)
    """
    def __init__(self, csv_path: str, root: str, clip_preprocess, max_samples: int = None, return_raw_pil=False):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42).reset_index(drop=True)
        self.root = Path(root)
        self.preprocess = clip_preprocess
        self.return_raw_pil = return_raw_pil  # if True, returns raw PIL as well for augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = str(row["image_path"])
        img_path = (self.root / img_rel) if not img_rel.startswith("poisoned/") else (self.root / img_rel)  # user root should contain poisoned/ if applicable
        img = Image.open(img_path).convert("RGB")
        img_t = self.preprocess(img)  # normalized tensor according to CLIP preprocess
        caption = str(row.get("caption", "")).strip()
        if not caption:
            caption = "unknown"
        poisoned_from = str(row.get("poisoned_from", "")) if "poisoned_from" in row.index else ""
        if self.return_raw_pil:
            return img_t, caption, poisoned_from, img_rel, img  # return original PIL for augmentation if needed
        return img_t, caption, poisoned_from, img_rel

# --------------------------
# Image augmentations for SimCLR views (operate on PIL.Image)
# --------------------------
def create_simclr_augmentations(image_size: int = 224):
    """
    Return a function that produces two augmented tensors (already normalized by CLIP preprocess if desired).
    We'll create lightweight transforms returning PIL images; caller should apply clip preprocess after augmentation.
    """
    # strong augmentations (paper-style)
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * image_size) if int(0.1 * image_size) % 2 == 1 else int(0.1 * image_size) + 1, sigma=(0.1, 2.0)),
    ])
    def two_augmentations(pil_image: Image.Image):
        return aug_transform(pil_image), aug_transform(pil_image)
    return two_augmentations

# --------------------------
# Evaluation helpers
# --------------------------
def build_candidate_texts(train_csv: str, val_csv: str, extra_targets: List[str] = None, exclude: List[str] = None) -> List[str]:
    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_csv)
    labels = sorted(set(df1['caption'].tolist() + df2['caption'].tolist()))
    labels = [l for l in labels if str(l).strip()]
    if extra_targets:
        for t in extra_targets:
            if t not in labels:
                labels.append(t)
    if exclude:
        labels = [l for l in labels if l not in set(exclude)]
    return labels

def evaluate_zero_shot(model, device, dataloader, candidates, clamp_logit_to=100.0):
    model.eval()
    preds = []
    with torch.no_grad():
        text_tokens = safe_tokenize(candidates, truncate=True, device=device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

    total = 0
    correct = 0
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

def compute_asr_from_preds(preds: List[Tuple[str, str, str, str]], attacker_target: str) -> float:
    """
    preds: list of tuples (image_path, gold_caption, pred_caption, poisoned_from)
    attacker_target: e.g. 'banana' - the label attacker wants
    ASR = fraction of poisoned images predicted as attacker_target
    """
    poisoned = [p for p in preds if (str(p[3]).strip() and str(p[3]).lower() != 'nan')]
    # also consider when image_path itself begins with 'poisoned/'
    poisoned += [p for p in preds if str(p[0]).startswith('poisoned/')]
    # unique-ify by image_path
    poisoned_map = {}
    for p in poisoned:
        poisoned_map[p[0]] = p
    poisoned = list(poisoned_map.values())
    if len(poisoned) == 0:
        return float('nan')
    attacker_hits = sum(1 for p in poisoned if p[2] == attacker_target)
    return attacker_hits / len(poisoned)

# --------------------------
# Training function (CleanCLIP finetune)
# --------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    # load CLIP
    model, clip_preprocess = clip.load(args.clip_backbone, device=device, jit=False)
    model.to(device)

    # clamp logit_scale initially to prevent runaway
    with torch.no_grad():
        model.logit_scale.data = torch.tensor(math.log(args.init_logit_scale), device=model.logit_scale.data.device)
    model.logit_scale.requires_grad = False
    print(f"Initial logit scale clamped to exp={float(model.logit_scale.exp().item()):.4f}")

    # Freeze text encoder for stability if desired
    if args.freeze_text:
        print("Freezing non-visual parameters (text encoder).")
        for name, p in model.named_parameters():
            if not name.startswith("visual"):
                p.requires_grad = False

    # dataset paths
    data_root = Path(args.data_root)
    train_csv = data_root / args.train_csv
    val_csv = data_root / args.val_csv
    assert train_csv.exists() and val_csv.exists(), f"Missing CSVs at {train_csv} or {val_csv}"

    # DataLoaders
    train_dataset = ImageTextCSV(str(train_csv), str(data_root), clip_preprocess, max_samples=args.max_samples_train, return_raw_pil=True)
    val_dataset = ImageTextCSV(str(val_csv), str(data_root), clip_preprocess, max_samples=args.max_samples_val, return_raw_pil=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate
    )

    # optimizer for trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # build candidate set for zero-shot evaluation (include attacker label to compute ASR)
    candidates = build_candidate_texts(str(train_csv), str(val_csv), extra_targets=[args.asr_target] if args.asr_target else None)
    print(f"Candidates count: {len(candidates)}")

    # augmentation helper for SimCLR-style views (returns PIL images)
    simclr_augment = create_simclr_augmentations(image_size=args.image_size)

    ce_loss = nn.CrossEntropyLoss()

    print("Beginning training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for step, batch in enumerate(pbar, start=1):
            # batch returns: img_tensor (CLIP preprocessed), caption, poisoned_from, img_rel, raw_pil
            imgs_t, captions, poisoned_froms, img_paths, raw_pils = batch
            imgs_t = imgs_t.to(device)  # CLIP-preprocessed view (not used for SimCLR augmentation)
            texts = list(captions)

            try:
                # Text tokens
                text_tokens = safe_tokenize(texts, truncate=True, device=device)

                # Encode primary features
                img_feats = model.encode_image(imgs_t)
                txt_feats = model.encode_text(text_tokens)

                if torch.isnan(img_feats).any() or torch.isnan(txt_feats).any():
                    print(f"Warning: NaN in base features at step {step}, skipping batch")
                    continue

                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # Multimodal CLIP loss (cross entropy on image->text and text->image logits)
                logit_scale = torch.clamp(model.logit_scale.exp(), max=args.clamp_logit_to)
                logits_per_image = logit_scale * (img_feats @ txt_feats.t())
                logits_per_text = logits_per_image.t()
                logits_per_image = torch.clamp(logits_per_image, min=-50.0, max=50.0)
                logits_per_text = torch.clamp(logits_per_text, min=-50.0, max=50.0)
                targets = torch.arange(len(imgs_t), device=device)
                loss_mm = (ce_loss(logits_per_image, targets) + ce_loss(logits_per_text, targets)) / 2.0
                loss = loss_mm

                # Unimodal image NT-Xent (SimCLR style) using two strong augmentations per sample
                if args.lambda_unimodal > 0.0:
                    # create two augmented views and pass through CLIP's preprocess & encoder
                    aug_imgs_a = []
                    aug_imgs_b = []
                    for pil in raw_pils:
                        a, b = simclr_augment(pil)
                        # apply CLIP preprocess to augmented PILs
                        ta = clip_preprocess(a)
                        tb = clip_preprocess(b)
                        aug_imgs_a.append(ta)
                        aug_imgs_b.append(tb)
                    aug_a = torch.stack(aug_imgs_a, dim=0).to(device)
                    aug_b = torch.stack(aug_imgs_b, dim=0).to(device)
                    with torch.no_grad() if args.no_grad_unimodal_forward else torch.enable_grad():
                        img_feats_a = model.encode_image(aug_a)
                        img_feats_b = model.encode_image(aug_b)
                    if (not torch.isnan(img_feats_a).any()) and (not torch.isnan(img_feats_b).any()):
                        img_feats_a = img_feats_a / (img_feats_a.norm(dim=-1, keepdim=True) + 1e-12)
                        img_feats_b = img_feats_b / (img_feats_b.norm(dim=-1, keepdim=True) + 1e-12)
                        loss_img_uni = nt_xent_loss(img_feats_a, img_feats_b, temperature=args.uni_temp)
                        loss = loss + args.lambda_unimodal * loss_img_uni

                # Unimodal text NT-Xent: produce two text views by masking or dropout
                if args.lambda_text_unimodal > 0.0:
                    try:
                        token_ids = clip.tokenize(texts, truncate=True).to(device)
                        # produce token-augmented view by random token masking
                        token_aug = token_ids.clone()
                        if args.text_aug_mode == "mask":
                            mask = (torch.rand(token_aug.shape, device=device) < args.text_mask_prob) & (token_aug != 0)
                            # replace masked tokens with zero (or choose a mask token if known)
                            token_aug[mask] = 0
                            with torch.no_grad():
                                txt_feats_aug = model.encode_text(token_aug)
                        else:
                            # dropout-based approach: set model to train + no_grad to simulate dropout, encode twice
                            was_train = model.training
                            model.train()
                            with torch.no_grad():
                                txt_feats_aug = model.encode_text(token_ids)
                            if not was_train:
                                model.eval()
                        if not torch.isnan(txt_feats_aug).any():
                            txt_feats_aug = txt_feats_aug / (txt_feats_aug.norm(dim=-1, keepdim=True) + 1e-12)
                            loss_text_uni = nt_xent_loss(txt_feats, txt_feats_aug, temperature=args.uni_temp)
                            loss = loss + args.lambda_text_unimodal * loss_text_uni
                    except Exception as e:
                        # fallback: small gaussian noise on text features
                        with torch.no_grad():
                            txt_feats_aug = txt_feats + 1e-3 * torch.randn_like(txt_feats)
                        txt_feats_aug = txt_feats_aug / (txt_feats_aug.norm(dim=-1, keepdim=True) + 1e-12)
                        loss_text_uni = nt_xent_loss(txt_feats, txt_feats_aug, temperature=args.uni_temp)
                        loss = loss + args.lambda_text_unimodal * loss_text_uni

                # numeric guards
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at step {step}, skipping")
                    continue

                opt.zero_grad()
                loss.backward()

                # gradient checks
                bad_grad = False
                for name, p in model.named_parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print(f"Warning: bad grad in {name}")
                        bad_grad = True
                        break
                if bad_grad:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
                opt.step()

                total_loss += float(loss.item()) * imgs_t.size(0)
                n_samples += imgs_t.size(0)
                pbar.set_postfix(loss=total_loss / max(1, n_samples))
            except Exception as e:
                print(f"ERROR batch {step}: {e}")
                continue

        avg_loss = total_loss / max(1, n_samples)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}")

        # checkpoint
        ckpt_name = f"{args.save_path}.epoch{epoch}.pt"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': opt.state_dict()}, ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        # quick eval on val
        small_val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        acc, preds = evaluate_zero_shot(model, device, small_val_loader, candidates, clamp_logit_to=args.clamp_logit_to)
        print(f"[Epoch {epoch}] Validation Acc (zero-shot): {acc*100:.2f}%")

        # save preds
        try:
            out_preds = Path(args.save_path).parent / f"preds_epoch{epoch}.csv"
            pd.DataFrame(preds, columns=["image_path", "gold_caption", "pred_caption", "poisoned_from"]).to_csv(out_preds, index=False)
            print("Saved epoch preds to", out_preds)
            # compute ASR if requested
            if args.asr_target:
                asr = compute_asr_from_preds(preds, args.asr_target)
                print(f"[Epoch {epoch}] ASR (target='{args.asr_target}'): {asr if not math.isnan(asr) else 'N/A'}")
        except Exception:
            pass

    print("Primary training phase complete.")

    # Optional supervised vision-only fine-tuning phase
    if args.vision_finetune:
        print("Starting supervised vision-only fine-tune phase.")
        # build label mapping
        df_train = pd.read_csv(train_csv)
        labels = sorted(set(df_train['caption'].tolist()))
        labels = [l for l in labels if str(l).strip()]
        label2idx = {l: i for i, l in enumerate(labels)}
        num_classes = len(labels)
        print(f"Vision fine-tune with {num_classes} classes.")

        # freeze CLIP
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        head = nn.Linear(model.visual.output_dim, num_classes).to(device)
        vf_opt = optim.AdamW(head.parameters(), lr=args.vf_lr, weight_decay=1e-4)
        vf_ce = nn.CrossEntropyLoss()

        def vf_collate(batch):
            imgs_t, caps, _, paths = zip(*batch)
            imgs = torch.stack(imgs_t, dim=0)
            labels_idx = torch.tensor([label2idx.get(c, 0) for c in caps], dtype=torch.long)
            return imgs, labels_idx

        vf_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=vf_collate)
        vf_val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=vf_collate)

        for e in range(1, args.vf_epochs + 1):
            head.train()
            total = 0.0; n = 0
            pbar = tqdm(vf_train_loader, desc=f"VF Epoch {e}")
            for imgs, labels_idx in pbar:
                imgs = imgs.to(device); labels_idx = labels_idx.to(device)
                with torch.no_grad():
                    img_feats = model.encode_image(imgs)
                    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                logits = head(img_feats)
                loss = vf_ce(logits, labels_idx)
                vf_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                vf_opt.step()
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

        vf_ckpt = f"{args.save_path}.vision_finetune.pt"
        torch.save({'head_state': head.state_dict(), 'label_map': labels}, vf_ckpt)
        print("Saved vision-finetune head to", vf_ckpt)

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="catsdogs_dataset")
    p.add_argument("--train-csv", default="train.csv")
    p.add_argument("--val-csv", default="val.csv")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--vf-lr", type=float, default=1e-4, help="vision fine-tune lr for linear head")
    p.add_argument("--vf-epochs", type=int, default=3)
    p.add_argument("--vision-finetune", action="store_true", help="run supervised vision-only finetune after main phase")
    p.add_argument("--device", default="cpu", help="cuda or cpu; default auto")
    p.add_argument("--save-path", default="cleanclip_full_ckpt.pt")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clip-backbone", default="ViT-B/32")
    p.add_argument("--init-logit-scale", dest="init_logit_scale", type=float, default=10.0)

    # CleanCLIP options
    p.add_argument("--clean", action="store_true", help="enable CleanCLIP style unimodal losses (lambda_unimodal and lambda_text_unimodal > 0)")
    p.add_argument("--lambda-unimodal", dest="lambda_unimodal", type=float, default=0.5, help="weight for image unimodal NT-Xent")
    p.add_argument("--lambda-text-unimodal", dest="lambda_text_unimodal", type=float, default=0.1, help="weight for text unimodal NT-Xent")
    p.add_argument("--uni-temp", dest="uni_temp", type=float, default=0.07, help="temperature for unimodal NT-Xent")
    p.add_argument("--text-aug-mode", dest="text_aug_mode", choices=["mask", "dropout"], default="mask", help="text augmentation mode for unimodal text loss")
    p.add_argument("--text-mask-prob", dest="text_mask_prob", type=float, default=0.15)
    p.add_argument("--no-grad-unimodal-forward", dest="no_grad_unimodal_forward", action="store_true", help="run unimodal forward passes under no_grad to save memory (less accurate grads for unimodal parts)")

    p.add_argument("--clamp-logit-to", dest="clamp_logit_to", type=float, default=100.0)
    p.add_argument("--freeze-text", dest="freeze_text", type=lambda x: (str(x).lower() == "true"), default=True)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    p.add_argument("--max-samples-train", dest="max_samples_train", type=int, default=None)
    p.add_argument("--max-samples-val", dest="max_samples_val", type=int, default=None)
    p.add_argument("--target-label", type=str, default=None, help="optional extra target label to include in candidate set")
    p.add_argument("--asr-target", type=str, default=None, help="attacker target label (e.g. 'banana') to compute ASR")
    p.add_argument("--image-size", type=int, default=224)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # normalize some arg names
    if not hasattr(args, "clamp_logit_to") or args.clamp_logit_to is None:
        args.clamp_logit_to = 100.0

    # small normalization
    args.no_grad_unimodal_forward = getattr(args, "no_grad_unimodal_forward", False)

    # run
    train(args)
