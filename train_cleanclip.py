

import argparse
import math
import random
from pathlib import Path
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
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07, eps: float = 1e-6):
    """
    Numerically safer NT-Xent. Avoids -inf diag entries and stabilizes rows.
    """
    device = z1.device
    B = z1.size(0)
    if B == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if torch.isnan(z1).any() or torch.isnan(z2).any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B,2B)

    # avoid -inf entries: replace diagonal with a large negative finite number
    large_neg = -1e9
    diag_mask = torch.eye(2 * B, device=device).bool()
    sim_masked = sim.masked_fill(diag_mask, large_neg)

    # row-wise stabilization
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

# --------------------------
# Dataset
# --------------------------
class ImageTextCSV(Dataset):
    """
    Expects CSV columns: image_path, caption
    Optional column: poisoned_from
    Returning raw PIL when return_raw_pil=True
    """
    def __init__(self, csv_path: str, root: str, clip_preprocess, max_samples: int = None, return_raw_pil: bool = False):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42).reset_index(drop=True)
        self.root = Path(root)
        self.preprocess = clip_preprocess
        self.return_raw_pil = return_raw_pil

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = str(row["image_path"])
        img_path = (self.root / img_rel)
        pil = Image.open(img_path).convert("RGB")
        img_t = self.preprocess(pil)
        caption = str(row.get("caption", "")).strip()
        if not caption:
            caption = "unknown"
        poisoned_from = str(row.get("poisoned_from", "")) if "poisoned_from" in row.index else ""
        if self.return_raw_pil:
            return img_t, caption, poisoned_from, img_rel, pil
        return img_t, caption, poisoned_from, img_rel

# --------------------------
# Collate function to handle raw PILs
# --------------------------
def flexible_collate(batch):
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

# --------------------------
# SimCLR-style augmentations (operate on PIL)
# --------------------------
def create_simclr_augmentations(image_size: int = 224):
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # GaussianBlur requires odd kernel size
        transforms.GaussianBlur(kernel_size= int(0.1 * image_size) if int(0.1 * image_size) % 2 == 1 else int(0.1 * image_size) + 1, sigma=(0.1, 2.0)),
    ])
    def two_augmentations(pil_image: Image.Image):
        return aug_transform(pil_image), aug_transform(pil_image)
    return two_augmentations

# --------------------------
# Evaluation helpers
# --------------------------
def build_candidate_texts(train_csv: str, val_csv: str, extra_targets: List[str] = None) -> List[str]:
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
    poisoned = [p for p in preds if (str(p[3]).strip() and str(p[3]).lower() != 'nan')]
    poisoned += [p for p in preds if str(p[0]).startswith('poisoned/')]
    poisoned_map = {}
    for p in poisoned:
        poisoned_map[p[0]] = p
    poisoned = list(poisoned_map.values())
    if len(poisoned) == 0:
        return float('nan')
    attacker_hits = sum(1 for p in poisoned if p[2] == attacker_target)
    return attacker_hits / len(poisoned)

# --------------------------
# Training function
# --------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    # load CLIP
    model, clip_preprocess = clip.load(args.clip_backbone, device=device, jit=False)

    # Force float32 to avoid fp16 overflow issues on some GPUs / CLIP builds
    model = model.float()
    model.to(device)

    # clamp logit scale (initial)
    with torch.no_grad():
        model.logit_scale.data = torch.tensor(math.log(args.init_logit_scale), device=model.logit_scale.data.device)
    # keep requires_grad False unless you want it trainable
    model.logit_scale.requires_grad = False
    print(f"Initial logit scale clamped to exp={float(model.logit_scale.exp().item()):.4f}")

    # Freeze text encoder for stability (paper uses freeze)
    if args.freeze_text:
        for name, p in model.named_parameters():
            if not name.startswith("visual"):
                p.requires_grad = False
        print("Text encoder frozen.")

    data_root = Path(args.data_root)
    train_csv = data_root / args.train_csv
    val_csv = data_root / args.val_csv
    assert train_csv.exists() and val_csv.exists(), f"Missing CSVs at {train_csv} or {val_csv}"

    # datasets (return raw PIL for augmentation)
    train_dataset = ImageTextCSV(str(train_csv), str(data_root), clip_preprocess, max_samples=args.max_samples_train, return_raw_pil=True)
    val_dataset = ImageTextCSV(str(val_csv), str(data_root), clip_preprocess, max_samples=args.max_samples_val, return_raw_pil=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=flexible_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=flexible_collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # keep attacker target in candidates so ASR is measurable
    candidates = build_candidate_texts(str(train_csv), str(val_csv), extra_targets=[args.asr_target] if args.asr_target else None)
    print(f"Candidates ({len(candidates)}): first 10 -> {candidates[:10]}")

    simclr_aug = create_simclr_augmentations(image_size=args.image_size)
    ce_loss = nn.CrossEntropyLoss()

    # --------------------------
    # inspect batch helper (runs once on first batch to print diagnostics)
    # --------------------------
    def inspect_batch(raw_pils, imgs_t, captions, device, model, args):
        print("=== INSPECT BATCH ===")
        print("Batch sizes:", len(raw_pils), imgs_t.shape if isinstance(imgs_t, torch.Tensor) else None)
        bad_images = []
        for i, p in enumerate(raw_pils):
            try:
                _ = p.size
            except Exception as e:
                bad_images.append((i, str(e)))
        if bad_images:
            print("Bad images detected:", bad_images)
        imgs_t = imgs_t.to(device)
        try:
            with torch.no_grad():
                img_feats = model.encode_image(imgs_t)
                txt_tokens = safe_tokenize(list(captions), device=device)
                txt_feats = model.encode_text(txt_tokens)

                # Normalize before similarity check (cosine)
                img_feats_n = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                txt_feats_n = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-12)

                print("img_feats:", tuple(img_feats.shape), "nan?", torch.isnan(img_feats).any().item(), "inf?", torch.isinf(img_feats).any().item())
                print("txt_feats:", tuple(txt_feats.shape), "nan?", torch.isnan(txt_feats).any().item(), "inf?", torch.isinf(txt_feats).any().item())
                ls = float(torch.clamp(model.logit_scale.exp(), max=args.clamp_logit_to).item())
                print("logit_scale (exp clamped) =", ls)
                sims = torch.clamp((img_feats_n @ txt_feats_n.t()) * ls, min=-50.0, max=50.0)
                print("sims min/max:", float(sims.min()), float(sims.max()))
        except Exception as e:
            print("Error during forward inspect:", e)
        print("=====================")

    print("Start training loop...")
    did_inspect = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for step, batch in enumerate(pbar, start=1):
            # batch: imgs_t (tensor), captions (list), poisoned_froms (list), img_paths (list), raw_pils (list)
            if len(batch) == 5:
                imgs_t, captions, poisoned_froms, img_paths, raw_pils = batch
            else:
                # fallback if dataset returns 4-tuple
                imgs_t, captions, poisoned_froms, img_paths = batch
                raw_pils = [None] * len(img_paths)

            if not did_inspect:
                try:
                    inspect_batch(raw_pils, imgs_t, captions, device, model, args)
                except Exception as e:
                    print("Inspect raised:", e)
                did_inspect = True

            imgs_t = imgs_t.to(device)
            texts = list(captions)

            try:
                text_tokens = safe_tokenize(texts, truncate=True, device=device)

                # primary features
                img_feats = model.encode_image(imgs_t)
                txt_feats = model.encode_text(text_tokens)

                if torch.isnan(img_feats).any() or torch.isnan(txt_feats).any():
                    print(f"Warning NaN in features at step {step}, skipping")
                    continue

                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-12)

                logit_scale = torch.clamp(model.logit_scale.exp(), max=args.clamp_logit_to)
                logits_per_image = logit_scale * (img_feats @ txt_feats.t())
                logits_per_text = logits_per_image.t()
                logits_per_image = torch.clamp(logits_per_image, min=-50.0, max=50.0)
                logits_per_text = torch.clamp(logits_per_text, min=-50.0, max=50.0)
                targets = torch.arange(len(imgs_t), device=device)
                loss_mm = (ce_loss(logits_per_image, targets) + ce_loss(logits_per_text, targets)) / 2.0
                loss = loss_mm

                # image unimodal NT-Xent (SimCLR-style)
                if args.lambda_unimodal > 0.0:
                    aug_a = []
                    aug_b = []
                    for pil in raw_pils:
                        # if raw_pil is None, fall back to using already preprocessed images (deterministic)
                        if pil is None:
                            # reconstruct PIL from tensor is expensive; skip unimodal if raw not available
                            aug_a = aug_b = None
                            break
                        a_pil, b_pil = simclr_aug(pil)
                        aug_a.append(clip_preprocess(a_pil))
                        aug_b.append(clip_preprocess(b_pil))
                    if aug_a is not None and len(aug_a) > 0:
                        aug_a = torch.stack(aug_a, dim=0).to(device)
                        aug_b = torch.stack(aug_b, dim=0).to(device)
                        img_feats_a = model.encode_image(aug_a)
                        img_feats_b = model.encode_image(aug_b)
                        if (not torch.isnan(img_feats_a).any()) and (not torch.isnan(img_feats_b).any()):
                            img_feats_a = img_feats_a / (img_feats_a.norm(dim=-1, keepdim=True) + 1e-12)
                            img_feats_b = img_feats_b / (img_feats_b.norm(dim=-1, keepdim=True) + 1e-12)
                            loss_img_uni = nt_xent_loss(img_feats_a, img_feats_b, temperature=args.uni_temp)
                            loss = loss + args.lambda_unimodal * loss_img_uni

                # text unimodal NT-Xent: encode twice with dropout-enabled forward to create two stochastic views
                if args.lambda_text_unimodal > 0.0:
                    was_training = model.training
                    model.train()
                    with torch.enable_grad():
                        txt_feats_view1 = model.encode_text(text_tokens)
                    with torch.enable_grad():
                        txt_feats_view2 = model.encode_text(text_tokens)
                    if not was_training:
                        model.eval()
                    if (not torch.isnan(txt_feats_view1).any()) and (not torch.isnan(txt_feats_view2).any()):
                        txt_feats_view1 = txt_feats_view1 / (txt_feats_view1.norm(dim=-1, keepdim=True) + 1e-12)
                        txt_feats_view2 = txt_feats_view2 / (txt_feats_view2.norm(dim=-1, keepdim=True) + 1e-12)
                        loss_text_uni = nt_xent_loss(txt_feats_view1, txt_feats_view2, temperature=args.uni_temp)
                        loss = loss + args.lambda_text_unimodal * loss_text_uni

                # numeric guards
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping NaN/Inf loss at step {step}")
                    continue

                optimizer.zero_grad()
                loss.backward()

                # gradient check
                bad_grad = False
                for name, p in model.named_parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print(f"Bad grad in {name}, skipping step")
                        bad_grad = True
                        break
                if bad_grad:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
                optimizer.step()

                total_loss += float(loss.item()) * imgs_t.size(0)
                n += imgs_t.size(0)
                pbar.set_postfix(loss=total_loss / max(1, n))
            except Exception as e:
                print(f"ERROR batch {step}: {e}")
                continue

        avg_loss = total_loss / max(1, n)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}")

        # checkpoint
        ckpt_name = f"{args.save_path}.epoch{epoch}.pt"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        # evaluation
        small_val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=flexible_collate)
        acc, preds = evaluate_zero_shot(model, device, small_val_loader, candidates, clamp_logit_to=args.clamp_logit_to)
        print(f"[Epoch {epoch}] Validation Acc (zero-shot): {acc*100:.2f}%")

        # save preds & compute ASR
        try:
            out_preds = Path(args.save_path).parent / f"preds_epoch{epoch}.csv"
            pd.DataFrame(preds, columns=["image_path", "gold_caption", "pred_caption", "poisoned_from"]).to_csv(out_preds, index=False)
            print("Saved epoch preds to", out_preds)
            if args.asr_target:
                asr = compute_asr_from_preds(preds, args.asr_target)
                print(f"[Epoch {epoch}] ASR (target='{args.asr_target}'): {asr if not math.isnan(asr) else 'N/A'}")
        except Exception:
            pass

    print("Primary training phase complete.")

    # optional supervised vision-only fine-tune (linear head)
    if args.vision_finetune:
        print("Starting supervised vision-only fine-tune phase.")
        df_train = pd.read_csv(train_csv)
        labels = sorted(set(df_train['caption'].tolist()))
        labels = [l for l in labels if str(l).strip()]
        label2idx = {l: i for i, l in enumerate(labels)}
        num_classes = len(labels)
        print(f"Vision fine-tune with {num_classes} classes.")

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
            total = 0.0; n_v = 0
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
                n_v += imgs.size(0)
                pbar.set_postfix(loss=total / max(1, n_v))
            print(f"[VF Epoch {e}] train_loss={total/max(1,n_v):.6f}")

            # VF validation
            head.eval()
            correct = 0; total_val = 0
            with torch.no_grad():
                for imgs, labels_idx in vf_val_loader:
                    imgs = imgs.to(device); labels_idx = labels_idx.to(device)
                    img_feats = model.encode_image(imgs)
                    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                    logits = head(img_feats)
                    preds_idx = logits.argmax(dim=1)
                    correct += (preds_idx == labels_idx).sum().item()
                    total_val += imgs.size(0)
            acc_v = correct / total_val if total_val > 0 else 0.0
            print(f"[VF Epoch {e}] VF Val Acc: {acc_v*100:.2f}%")

        vf_ckpt = f"{args.save_path}.vision_finetune.pt"
        torch.save({'head_state': head.state_dict(), 'label_map': labels}, vf_ckpt)
        print("Saved vision-finetune head to", vf_ckpt)

# --------------------------
# CLI
# --------------------------
def parse_args():
    """
    Parses command-line arguments for CleanCLIP training.
    Colab-safe (ignores extra Jupyter args).
    """
    parser = argparse.ArgumentParser(
        description="Train CleanCLIP with configurable loss weights and hyperparameters."
    )

    # -------------------- Dataset & Paths --------------------
    parser.add_argument("--data-root", type=str, default="catsdogs_dataset",
                        help="Path to dataset root containing images/ and CSVs.")
    parser.add_argument("--train-csv", type=str, default="train.csv",
                        help="CSV file for training data (relative to data-root).")
    parser.add_argument("--val-csv", type=str, default="val.csv",
                        help="CSV file for validation data (relative to data-root).")
    parser.add_argument("--save-path", type=str, default="cleanclip_lambda1_ckpt.pt",
                        help="Path to save final model checkpoint.")

    # -------------------- Training Settings --------------------
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for main model.")
    parser.add_argument("--vf-lr", type=float, default=1e-5, help="Learning rate for vision finetuning.")
    parser.add_argument("--vf-epochs", type=int, default=3, help="Epochs for vision backbone finetuning.")
    parser.add_argument("--vision-finetune", action="store_true",
                        help="If set, enables vision encoder fine-tuning.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu). Default: auto-detect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # -------------------- Model Configuration --------------------
    parser.add_argument("--clip-backbone", type=str, default="ViT-B/32",
                        help="CLIP backbone architecture (e.g., ViT-B/32, ViT-L/14).")
    parser.add_argument("--init-logit-scale", type=float, default=10.0,
                        help="Initial logit scale before training.")

    # -------------------- Loss Weights (λ) --------------------
    parser.add_argument("--lambda-unimodal", type=float, default=0.1,
                        help="Weight for image unimodal NT-Xent loss.")
    parser.add_argument("--lambda-text-unimodal", type=float, default=0.1,
                        help="Weight for text unimodal NT-Xent loss.")
    parser.add_argument("--uni-temp", type=float, default=0.07,
                        help="Temperature for unimodal NT-Xent.")

    # -------------------- Text Augmentation --------------------
    parser.add_argument("--text-aug-mode", choices=["mask", "dropout"], default="dropout",
                        help="Text augmentation type: mask or dropout.")
    parser.add_argument("--text-mask-prob", type=float, default=0.15,
                        help="Mask probability for text augmentation when mode='mask'.")

    # -------------------- Attack / Poison Analysis --------------------
    parser.add_argument("--asr-target", type=str, default="banana",
                        help="Target label for computing attack success rate (ASR).")
    parser.add_argument("--target-label", type=str, default=None,
                        help="Extra target label to include in candidate set.")

    # -------------------- Regularization & Optimizer --------------------
    parser.add_argument("--clamp-logit-to", type=float, default=100.0,
                        help="Clamp logits to avoid overflow in softmax.")
    parser.add_argument("--freeze-text", type=lambda x: str(x).lower() == "true", default=True,
                        help="Freeze text encoder parameters (default: True).")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping threshold.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for optimizer.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of DataLoader workers.")

    # -------------------- Data Limits & Image Size --------------------
    parser.add_argument("--max-samples-train", type=int, default=None,
                        help="Optional max samples for training (debug mode).")
    parser.add_argument("--max-samples-val", type=int, default=None,
                        help="Optional max samples for validation (debug mode).")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size for model.")

    # Colab-safe parse (ignore unknown args like -f /kernel-XXXX.json)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # normalization fallback
    if not hasattr(args, "clamp_logit_to") or args.clamp_logit_to is None:
        args.clamp_logit_to = 100.0

    train(args)
