#!/usr/bin/env python3
"""
train_cleanclip_filtered_final.py

Robust CleanCLIP fine-tuning on filtered (clean-only) training data.

Usage example:
  python train_cleanclip_filtered_final.py \
    --data-root catsdogs_dataset \
    --train-csv catsdogs_dataset/train.csv \
    --val-csv catsdogs_dataset/val.csv \
    --checkpoint ./finetuneclip_ckpt/finetune_epoch5.pt \
    --save-path ./cleanclip_filtered_final_ckpt \
    --epochs 3 \
    --batch-size 16 \
    --lr 1e-6 \
    --lambda1 1.0 \
    --lambda2 0.5 \
    --asr-target "a photo of a banana" \
    --freeze-logit
"""

import os
import csv
import argparse
import traceback
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import clip

# optional: torchvision for saving preview images
try:
    import torchvision
    _TORCHVISION_AVAILABLE = True
except Exception:
    _TORCHVISION_AVAILABLE = False

# -------------------------
# Helpers
# -------------------------
def norm_caption(s):
    return " ".join(s.lower().strip().split()) if s is not None else ""

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_bad_sample(save_path, tag, epoch, batch_idx, sample_idx, path, caption, reason, tensor=None):
    ensure_dir(save_path)
    line = f"{tag} epoch={epoch} batch={batch_idx} idx={sample_idx} path={path} reason={reason} caption={caption}\n"
    with open(os.path.join(save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
        fo.write(line)
    try:
        if _TORCHVISION_AVAILABLE and tensor is not None:
            t = tensor.clone().detach().cpu()
            if t.dim() == 3:
                t = t.unsqueeze(0)
            preview = os.path.join(save_path, f"bad_{tag}_e{epoch}_b{batch_idx}_i{sample_idx}.png")
            torchvision.utils.save_image(t, preview, normalize=True)
    except Exception as e:
        with open(os.path.join(save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
            fo.write(f"preview_save_failed: {e}\n")

def safe_preprocess(path, preprocess):
    """Return (tensor, err). tensor is CxHxW float or None on error."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return None, f"PIL_open_error:{e}"
    try:
        t = preprocess(img)  # usually float tensor
    except Exception as e:
        return None, f"preprocess_error:{e}"
    if torch.isnan(t).any().item() or torch.isinf(t).any().item():
        return None, "tensor_nan_or_inf"
    vmin = float(t.min().item()); vmax = float(t.max().item())
    if vmin < -1e3 or vmax > 1e3:
        return None, f"tensor_out_of_range min={vmin:.2f},max={vmax:.2f}"
    return t, None

# -------------------------
# Dataset index-only (preprocess in loop for robustness)
# -------------------------
class CleanIndexDataset(Dataset):
    def __init__(self, csv_file, data_root, skip_predicates=None):
        self.samples = []
        self.data_root = data_root
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "caption" not in reader.fieldnames:
                raise ValueError("CSV must contain 'image_path' and 'caption' columns")
            for row in reader:
                rel = row["image_path"]
                rel_n = rel.replace("\\", os.path.sep).replace("/", os.path.sep).lstrip(os.path.sep)
                path = os.path.join(data_root, rel_n)
                caption = row["caption"]
                skip = False
                if skip_predicates:
                    for pred in skip_predicates:
                        if pred(path, caption):
                            skip = True
                            break
                if not skip:
                    self.samples.append((path, caption))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------------
# CleanCLIP loss: L_CLIP + L_SS
# -------------------------
def compute_cleanclip_loss(model, frozen_model, images, tokens, device, lambda1, lambda2):
    # get features and force FP32 (avoid fp16 overflow)
    img_feat = model.encode_image(images).float()
    txt_feat = model.encode_text(tokens).float()
    # normalize (with eps)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-10)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-10)

    # CLIP loss (symmetric)
    logit_scale = model.logit_scale.exp().to(dtype=img_feat.dtype)
    logits_img = logit_scale * (img_feat @ txt_feat.t())
    logits_txt = logits_img.t()
    labels = torch.arange(len(images), device=device)
    loss_clip = (torch.nn.functional.cross_entropy(logits_img, labels) +
                 torch.nn.functional.cross_entropy(logits_txt, labels)) / 2.0

    # Stability/self-supervised loss relative to frozen model (no grad)
    with torch.no_grad():
        frozen_img = frozen_model.encode_image(images).float()
        frozen_txt = frozen_model.encode_text(tokens).float()
        frozen_img = frozen_img / (frozen_img.norm(dim=-1, keepdim=True) + 1e-10)
        frozen_txt = frozen_txt / (frozen_txt.norm(dim=-1, keepdim=True) + 1e-10)
    loss_ss = (torch.nn.functional.mse_loss(img_feat, frozen_img) +
               torch.nn.functional.mse_loss(txt_feat, frozen_txt)) / 2.0

    total = lambda1 * loss_clip + lambda2 * loss_ss
    return total, float(loss_clip.detach().cpu().item()), float(loss_ss.detach().cpu().item())

# -------------------------
# Evaluation and ASR helpers
# -------------------------
def evaluate_and_save_preds(model, val_index_ds, preprocess, device, save_path, epoch, eval_batch_size=64):
    model.eval()
    total_correct = 0
    total_samples = 0
    pred_rows = []
    val_loader = DataLoader(list(range(len(val_index_ds))), batch_size=eval_batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_indices in val_loader:
            imgs = []
            caps = []
            paths = []
            for ds_idx in batch_indices:
                p, c = val_index_ds[ds_idx]
                t, err = safe_preprocess(p, preprocess)
                if err:
                    log_bad_sample(save_path, "val", epoch, 0, ds_idx, p, c, "val_preprocess_err:"+err, tensor=t)
                    continue
                imgs.append(t)
                caps.append(c)
                paths.append(p)
            if len(imgs) == 0:
                continue

            images = torch.stack(imgs, dim=0).to(device)
            tokens = clip.tokenize(caps, truncate=True).to(device)

            # clamp logit_scale
            with torch.no_grad():
                model.logit_scale.data = torch.clamp(model.logit_scale.data, min=-5.0, max=4.0)

            try:
                image_features = model.encode_image(images).float()
                text_features = model.encode_text(tokens).float()
            except Exception as e:
                with open(os.path.join(save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"eval epoch={epoch} encode_exception: {e}\n")
                continue

            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-10)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
            logit_scale = model.logit_scale.exp().to(dtype=image_features.dtype)
            sims = (logit_scale * (image_features @ text_features.t()))
            preds = sims.argmax(dim=1).tolist()

            for i, pred_idx in enumerate(preds):
                pred_caption = caps[pred_idx]
                gold = caps[i]
                correct = int(norm_caption(pred_caption) == norm_caption(gold))
                pred_rows.append({
                    "image_path": paths[i],
                    "gold_caption": gold,
                    "pred_caption": pred_caption,
                    "correct": correct
                })
                total_correct += correct
                total_samples += 1

    acc = total_correct / max(1, total_samples)
    ensure_dir(save_path)
    csv_path = os.path.join(save_path, f"cleanclip_preds_epoch{epoch}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "gold_caption", "pred_caption", "correct"])
        writer.writeheader()
        writer.writerows(pred_rows)
    return acc, csv_path, pred_rows

def compute_asr_from_preds(pred_rows, val_csv, data_root, save_path, target):
    pred_map = {}
    for r in pred_rows:
        p = r["image_path"].replace("\\", os.path.sep).replace("/", os.path.sep)
        pred_map[p] = r["pred_caption"]

    val_meta = []
    with open(val_csv, newline='', encoding='utf-8') as vf:
        vr = csv.DictReader(vf)
        for r in vr:
            rel = r.get("image_path", "")
            rel_n = rel.replace("\\", os.path.sep).replace("/", os.path.sep).lstrip(os.path.sep)
            fullp = os.path.join(data_root, rel_n)
            is_poison = False
            for key in ["poisoned","poisoned_from","is_poison","poison","poison_flag"]:
                if key in r and str(r.get(key)).strip().lower() not in ["", "nan", "none", "false", "0"]:
                    is_poison = True
                    break
            if not is_poison and "poison" in rel_n.lower():
                is_poison = True
            val_meta.append({"path": fullp, "caption": r.get("caption",""), "poisoned": is_poison})

    target_norm = norm_caption(target)
    poisoned_entries = []
    for vm in val_meta:
        if not vm["poisoned"]:
            continue
        pred_caption = pred_map.get(vm["path"], "")
        is_target = int(norm_caption(pred_caption) == target_norm)
        poisoned_entries.append({
            "image_path": vm["path"],
            "gold_caption": vm["caption"],
            "pred_caption": pred_caption,
            "is_target": is_target
        })

    total_poison = len(poisoned_entries)
    total_success = sum([e["is_target"] for e in poisoned_entries])
    asr = (total_success / total_poison) if total_poison>0 else float("nan")
    out_csv = os.path.join(save_path, f"cleanclip_asr_details.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as af:
        writer = csv.DictWriter(af, fieldnames=["image_path","gold_caption","pred_caption","is_target"])
        writer.writeheader()
        writer.writerows(poisoned_entries)
    return asr, total_success, total_poison, out_csv

# -------------------------
# Main
# -------------------------
import copy
import time
import csv

def run_ablation(base_args):
    """
    Run ablation study over loss weights:
      (1) Default: lambda1=1.0, lambda2=base_args.lambda2
      (2) Only stability loss: lambda1=0.0, lambda2=base_args.lambda2
      (3) Only CLIP loss: lambda1=1.0, lambda2=0.0

    Saves results to ablation_results.csv under base_args.save_path.
    """

    ablation_settings = [
        {"name": "baseline", "lambda1": base_args.lambda1, "lambda2": base_args.lambda2},
        {"name": "stability_only", "lambda1": 0.0, "lambda2": base_args.lambda2},
        {"name": "clip_only", "lambda1": base_args.lambda1, "lambda2": 0.0},
    ]

    results = []
    csv_path = os.path.join(base_args.save_path, "ablation_results.csv")
    ensure_dir(base_args.save_path)

    for config in ablation_settings:
        print("\n==============================")
        print(f" Running Ablation: {config['name']} ")
        print(f" lambda1={config['lambda1']}  lambda2={config['lambda2']}")
        print("==============================\n")

        args = copy.deepcopy(base_args)
        args.lambda1 = config["lambda1"]
        args.lambda2 = config["lambda2"]

        # give each run its own subfolder
        args.save_path = os.path.join(base_args.save_path, f"ablation_{config['name']}")
        ensure_dir(args.save_path)

        start_time = time.time()
        try:
            main(args)
        except Exception as e:
            print(f"⚠️  Ablation '{config['name']}' failed: {e}")
            continue
        end_time = time.time()
        elapsed = round(end_time - start_time, 2)

        # read last preds csv and asr file if exist
        last_pred_csv = None
        last_asr_csv = None
        acc = asr = succ = tot = float("nan")

        # try to find last epoch file
        all_files = os.listdir(args.save_path)
        epoch_nums = []
        for fn in all_files:
            if fn.startswith("cleanclip_preds_epoch") and fn.endswith(".csv"):
                try:
                    ep = int(fn.replace("cleanclip_preds_epoch", "").replace(".csv", ""))
                    epoch_nums.append(ep)
                except:
                    pass
        if epoch_nums:
            last_ep = max(epoch_nums)
            last_pred_csv = os.path.join(args.save_path, f"cleanclip_preds_epoch{last_ep}.csv")
            last_asr_csv = os.path.join(args.save_path, f"cleanclip_asr_details.csv")
            # parse val accuracy
            with open(last_pred_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                preds = list(reader)
            if preds:
                acc = sum(int(r["correct"]) for r in preds) / len(preds)
            # parse ASR file
            if os.path.exists(last_asr_csv):
                with open(last_asr_csv, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    asr_rows = list(reader)
                    if asr_rows:
                        succ = sum(int(r["is_target"]) for r in asr_rows)
                        tot = len(asr_rows)
                        asr = succ / tot if tot > 0 else float("nan")

        results.append({
            "name": config["name"],
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "val_accuracy": acc,
            "ASR": asr,
            "ASR_success": succ,
            "ASR_total": tot,
            "runtime_sec": elapsed,
            "pred_csv": last_pred_csv,
            "asr_csv": last_asr_csv,
        })

        print(f"✅ Done: {config['name']} | Acc={acc*100:.2f}% | ASR={asr*100:.2f}% | Time={elapsed}s")

    # save all results
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Ablation summary ===")
    for r in results:
        print(f"{r['name']}: acc={r['val_accuracy']*100:.2f}%  asr={r['ASR']*100:.2f}%  (λ1={r['lambda1']} λ2={r['lambda2']})")
    print(f"\nSaved all results to {csv_path}\n")

def main(args):
    # make sure save_path exists early (avoid FileNotFoundError on logging)
    ensure_dir(args.save_path)

    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"
    print("Device:", device)
    print("Save path:", args.save_path)

    # frozen reference model: pretrained CLIP
    frozen_model, preprocess = clip.load(args.clip_backbone, device=device, jit=False)
    frozen_model.eval()
    frozen_model.float()

    # trainable model: load pretrained and checkpoint into it
    model, _ = clip.load(args.clip_backbone, device=device, jit=False)
    # force FP32 params
    model.float()

    # reset/clamp logit_scale to safe starting value
    with torch.no_grad():
        model.logit_scale.data.fill_(0.0)   # exp(0)=1.0
        model.logit_scale.data.clamp_(-2.0, 2.0)

    # optionally load checkpoint (partial allowed)
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print("Loading checkpoint into model:", args.checkpoint)
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state, strict=False)
        else:
            print("Checkpoint path provided but not found:", args.checkpoint)

    model.to(device)
    model.train()

    # decide optimizer param groups: optionally freeze logit_scale or give tiny lr
    if args.freeze_logit:
        for n,p in model.named_parameters():
            if "logit_scale" in n:
                p.requires_grad = False
        opt_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        # separate group for logit_scale with tiny LR
        logit_params = [p for n,p in model.named_parameters() if "logit_scale" in n and p.requires_grad]
        other_params = [p for n,p in model.named_parameters() if "logit_scale" not in n and p.requires_grad]
        groups = [{"params": other_params, "lr": args.lr}, {"params": logit_params, "lr": args.logit_lr}]
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # build filtered train index dataset (skip "poison", "banana", "sketch")
    def skip_pred(path, caption):
        lo_caption = (caption or "").lower()
        lo_path = (path or "").lower()
        if "poison" in lo_path: return True
        if "banana" in lo_caption: return True
        if "sketch" in lo_caption: return True
        return False

    train_index_ds = CleanIndexDataset(args.train_csv, args.data_root, skip_predicates=[skip_pred])
    val_index_ds = CleanIndexDataset(args.val_csv, args.data_root, skip_predicates=[])  # keep val full

    print("Filtered train samples:", len(train_index_ds))   

    # loaders built as list-of-indices and preprocessed inside loop for robustness
    train_loader = DataLoader(list(range(len(train_index_ds))), batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        total_clip = 0.0
        total_ss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch_indices in enumerate(pbar):
            imgs = []
            caps = []
            paths = []
            for local_idx, ds_idx in enumerate(batch_indices):
                path, caption = train_index_ds[ds_idx]
                t, err = safe_preprocess(path, preprocess)
                if err:
                    log_bad_sample(args.save_path, "train", epoch, batch_idx, local_idx, path, caption, "preprocess_err:"+err, tensor=t)
                    continue
                imgs.append(t)
                caps.append(caption)
                paths.append(path)
            if len(imgs) == 0:
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch} batch={batch_idx} all_bad_samples_skipped\n")
                continue

            images = torch.stack(imgs, dim=0).to(device)
            tokens = clip.tokenize(caps, truncate=True).to(device)

            # clamp logit_scale per batch
            with torch.no_grad():
                model.logit_scale.data = torch.clamp(model.logit_scale.data, min=-5.0, max=4.0)

            # compute loss robustly
            try:
                loss, loss_clip_val, loss_ss_val = compute_cleanclip_loss(model, frozen_model, images, tokens, device, args.lambda1, args.lambda2)
            except Exception as e:
                tb = traceback.format_exc()
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch} batch={batch_idx} forward_exception: {e}\n{tb}\n")
                # isolate per-sample to find offending ones
                for i_sample, (pth, cap) in enumerate(zip(paths, caps)):
                    t_single, err2 = safe_preprocess(pth, preprocess)
                    if err2:
                        log_bad_sample(args.save_path, "train_isolate", epoch, batch_idx, i_sample, pth, cap, "preprocess_err:"+err2, tensor=t_single)
                        continue
                    try:
                        t_b = t_single.unsqueeze(0).to(device)
                        tok_b = clip.tokenize([cap]).to(device)
                        _loss, _, _ = compute_cleanclip_loss(model, frozen_model, t_b, tok_b, device, args.lambda1, args.lambda2)
                    except Exception as ee:
                        log_bad_sample(args.save_path, "train_isolate", epoch, batch_idx, i_sample, pth, cap, f"forward_exception:{ee}", tensor=t_single)
                continue

            # catch NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch} batch={batch_idx} loss_nan_or_inf - skipped\n")
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_clip += float(loss_clip_val)
            total_ss += float(loss_ss_val)
            steps += 1
            pbar.set_postfix(loss=f"{(total_loss/steps):.6f}")

        if steps > 0:
            print(f"Epoch {epoch} avg loss {total_loss/steps:.6f} (clip={total_clip/steps:.6f}, ss={total_ss/steps:.6f})")
        else:
            print(f"Epoch {epoch} had 0 successful steps.")

        # checkpoint
        ensure_dir(args.save_path)
        ckpt = os.path.join(args.save_path, f"cleanclip_filtered_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        print("Saved checkpoint:", ckpt)

        # evaluate + ASR
        acc, preds_csv, pred_rows = evaluate_and_save_preds(model, val_index_ds, preprocess, device, args.save_path, epoch, eval_batch_size=args.eval_batch_size)
        print(f"Validation accuracy (per-sample) after epoch {epoch}: {acc*100:.2f}%")
        asr, succ, tot, asr_csv = compute_asr_from_preds(pred_rows, args.val_csv, args.data_root, args.save_path, args.asr_target)
        print(f"ASR (target='{args.asr_target}'): {asr*100:.2f}% ({succ}/{tot})  details -> {asr_csv}")

    print("Training complete.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanCLIP filtered training (final, robust)")
    parser.add_argument("--data-root", type=str, default="catsdogs_dataset")
    parser.add_argument("--train-csv", type=str, default="catsdogs_dataset/train2.csv")
    parser.add_argument("--val-csv", type=str, default="catsdogs_dataset/val.csv")
    parser.add_argument("--checkpoint", type=str, default="finetuneclip_ckpt/finetune_epoch1.pt", help="Optional .pt checkpoint to load into trainable model")
    parser.add_argument("--save-path", type=str, default="./cleanclip_filtered_final_ckpt")
    parser.add_argument("--clip-backbone", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--logit-lr", type=float, default=1e-8, help="LR for logit_scale if not frozen")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=1.0, help="weight for CLIP loss")
    parser.add_argument("--lambda2", type=float, default=8, help="weight for stability (L_SS)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-logit",default=False, action="store_true", help="Freeze logit_scale parameter (recommended to avoid explosion)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--asr-target", type=str, default="This is a sketch of banana")
    args = parser.parse_args()

    # fallback for device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to cpu.")
        args.device = "cpu"

    main(args)
