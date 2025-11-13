#!/usr/bin/env python3
"""
final.py

Robust vanilla CLIP fine-tuning (FP32) with defensive checks:
 - clamps learned logit_scale each step
 - per-sample preprocessing checks (skip bad samples within a batch)
 - if a batch produces NaNs, isolates and logs offending samples, saves previews
 - gradient clipping and conservative default LR
 - saves per-epoch predictions CSV and checkpoints
"""

import os
import csv
import argparse
import sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import traceback

# torchvision optional (for saving image previews)
try:
    import torchvision
    _TORCHVISION_AVAILABLE = True
except Exception:
    _TORCHVISION_AVAILABLE = False

# --------------------------
# Dataset
# --------------------------
class CsvImageTextDataset(Dataset):
    def __init__(self, csv_file, data_root, preprocess):
        self.samples = []
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "caption" not in reader.fieldnames:
                raise ValueError(f"CSV {csv_file} must contain 'image_path' and 'caption' columns")
            for row in reader:
                relpath = row["image_path"]
                # normalize path separators: avoid leading slash problems
                relpath = relpath.replace("\\", os.path.sep).replace("/", os.path.sep).lstrip(os.path.sep)
                path = os.path.join(data_root, relpath)
                caption = row["caption"]
                self.samples.append((path, caption))
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        # leave actual PIL open to caller; here just return path+caption so caller can handle errors
        return path, caption


# --------------------------
# Utility: preprocess single image and detect issues
# --------------------------
def safe_preprocess(path, preprocess):
    """Return (tensor, err_str). tensor is a torch.Tensor or None if failed. err_str is None on success."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return None, f"PIL_open_error:{e}"
    try:
        t = preprocess(img)  # tensor CxHxW float
    except Exception as e:
        return None, f"preprocess_error:{e}"
    # check finiteness
    if torch.isnan(t).any().item() or torch.isinf(t).any().item():
        return None, "tensor_nan_or_inf"
    # sanity range check
    vmin = float(t.min().item()); vmax = float(t.max().item())
    if vmin < -1e3 or vmax > 1e3:
        return None, f"tensor_out_of_range min={vmin:.2f},max={vmax:.2f}"
    return t, None


# --------------------------
# Accuracy metric (in-batch)
# --------------------------
def compute_inbatch_accuracy(image_features, text_features):
    sims = image_features @ text_features.t()
    preds = sims.argmax(dim=1)
    labels = torch.arange(len(image_features), device=image_features.device)
    correct = (preds == labels).sum().item()
    return correct / len(labels), preds


# --------------------------
# Logging helpers
# --------------------------
def log_bad_sample(save_path, epoch, batch_idx, sample_idx, path, caption, reason, tensor=None):
    os.makedirs(save_path, exist_ok=True)
    log_line = f"epoch={epoch} batch={batch_idx} idx={sample_idx} path={path} reason={reason} caption={caption}\n"
    with open(os.path.join(save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
        fo.write(log_line)
    # save preview image if possible
    try:
        if _TORCHVISION_AVAILABLE and tensor is not None:
            # tensor is CxHxW or batched; ensure shape for save_image is (B,C,H,W)
            t = tensor.clone().detach().cpu()
            if t.dim() == 3:
                t = t.unsqueeze(0)
            preview_path = os.path.join(save_path, f"bad_epoch{epoch}_b{batch_idx}_i{sample_idx}.png")
            torchvision.utils.save_image(t, preview_path, normalize=True)
    except Exception as e:
        with open(os.path.join(save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
            fo.write(f"preview_save_failed: {e}\n")


# --------------------------
# Main training function
# --------------------------
# --------------------------
# ASR calculation block (place immediately after the validation/pred CSV save)
# Requires: args.asr_target (string). If not provided, ASR is skipped.
# --------------------------


def train(args):
    # device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device
    use_cuda = (device == "cuda")
    print(f"Using device: {device} (FP32). lr={args.lr} grad_clip={args.grad_clip}")

    # load CLIP
    model, preprocess = clip.load(args.clip_backbone, device="cpu", jit=False)  # load on cpu, then move
    model = model.to(device)

    # optional dataset check
    if args.check_data:
        print("Running quick dataset check (preprocess each file)...")
        ds = CsvImageTextDataset(args.train_csv, args.data_root, preprocess)
        bad_count = 0
        for i, (path, caption) in enumerate(ds):
            t, err = safe_preprocess(path, preprocess)
            if err:
                print(f"bad[{i}]: {path} -> {err}")
                bad_count += 1
                if bad_count >= args.max_report_bad:
                    break
        print(f"Found {bad_count} problematic samples (showing up to {args.max_report_bad}). Exiting check-data mode.")
        return

    # Build datasets (we return paths+captions; preprocess later with safety)
    train_index_ds = CsvImageTextDataset(args.train_csv, args.data_root, preprocess)
    val_index_ds = CsvImageTextDataset(args.val_csv, args.data_root, preprocess)

    # DataLoaders will yield indices from the dataset; we'll manually preprocess within loop to handle errors
    train_loader = DataLoader(list(range(len(train_index_ds))), batch_size=args.batch_size, shuffle=True,
                              num_workers=0)  # use 0 workers to simplify per-sample error handling
    val_loader = DataLoader(list(range(len(val_index_ds))), batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch_indices in enumerate(pbar):
            # build batch safely: preprocess each sample and skip bad ones
            imgs_tensors = []
            captions = []
            paths = []
            bad_in_batch = []
            for local_idx, ds_idx in enumerate(batch_indices):
                path, caption = train_index_ds[ds_idx]
                t, err = safe_preprocess(path, preprocess)
                if err:
                    bad_in_batch.append((local_idx, ds_idx, path, caption, err))
                    log_bad_sample(args.save_path, epoch+1, batch_idx, local_idx, path, caption, err, tensor=t)
                    continue
                imgs_tensors.append(t)
                captions.append(caption)
                paths.append(path)

            if len(imgs_tensors) == 0:
                # nothing usable in this batch, skip
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch+1} batch={batch_idx} all_samples_bad, skipped batch\n")
                continue

            # stack into batch tensor
            images = torch.stack(imgs_tensors, dim=0).to(device, non_blocking=True)
            tokens = clip.tokenize(captions, truncate=True).to(device)

            # clamp logit_scale raw param
            with torch.no_grad():
                model.logit_scale.data = torch.clamp(model.logit_scale.data, min=-5.0, max=4.0)

            # forward; guard with try/except to isolate NaN-producing samples
            try:
                image_features = model.encode_image(images)
                text_features = model.encode_text(tokens)
            except Exception as e:
                # forward raised; attempt per-sample isolate
                tb = traceback.format_exc()
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch+1} batch={batch_idx} forward_exception: {e}\n{tb}\n")
                # try to find which sample(s) fail when encoded individually
                for i_sample, (path, caption) in enumerate(zip(paths, captions)):
                    try:
                        t_single, _ = safe_preprocess(path, preprocess)
                        if t_single is None:
                            log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, "preprocess_failed_on_forwardcheck")
                            continue
                        t_single = t_single.unsqueeze(0).to(device)
                        # wrap in try
                        try:
                            _ = model.encode_image(t_single)
                        except Exception as e2:
                            log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, f"encode_image_failed:{e2}", tensor=t_single)
                            continue
                        # check text encoding
                        try:
                            tok = clip.tokenize([caption], truncate=True).to(device)
                            _ = model.encode_text(tok)
                        except Exception as e3:
                            log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, f"encode_text_failed:{e3}")
                            continue
                    except Exception as ee:
                        log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, f"isolation_failed:{ee}")
                # skip this batch (we logged offending samples)
                continue

            # ensure features finite
            if torch.isnan(image_features).any().item() or torch.isnan(text_features).any().item() \
               or torch.isinf(image_features).any().item() or torch.isinf(text_features).any().item():
                # try to isolate problematic samples by individual encoding
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch+1} batch={batch_idx} batch_features_nan_or_inf\n")
                # check per-sample
                for i_sample, (path, caption) in enumerate(zip(paths, captions)):
                    try:
                        t_single, err = safe_preprocess(path, preprocess)
                        if err:
                            log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, f"preprocess_err:{err}", tensor=t_single)
                            continue
                        t_singleb = t_single.unsqueeze(0).to(device)
                        try:
                            imf = model.encode_image(t_singleb)
                            txt = model.encode_text(clip.tokenize([caption], truncate=True).to(device))
                        except Exception as e2:
                            log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, f"encode_exception:{e2}", tensor=t_singleb)
                            continue
                        # check nan/inf on these single outputs
                        if torch.isnan(imf).any().item() or torch.isnan(txt).any().item() or torch.isinf(imf).any().item() or torch.isinf(txt).any().item():
                            log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, "single_feature_nan_or_inf", tensor=t_singleb)
                    except Exception as ee:
                        log_bad_sample(args.save_path, epoch+1, batch_idx, i_sample, path, caption, f"isolation_failed:{ee}")
                # skip the problematic batch after logging
                continue

            # normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # safe compute logit scale (float dtype)
            logit_scale = model.logit_scale.exp().to(dtype=image_features.dtype)
            logits_per_image = logit_scale * (image_features @ text_features.t())
            logits_per_text = logits_per_image.t()
            labels = torch.arange(len(images), device=device)

            # loss
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2.0

            # If loss is NaN here, log and skip updating
            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                    fo.write(f"epoch={epoch+1} batch={batch_idx} loss_nan_after_forward - skipping update\n")
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}: train loss = {avg_loss:.6f}")

        # --------------------------
        # Evaluate + Save Predictions (in-batch matching)
        # --------------------------
        model.eval()
        total_correct = 0
        total_samples = 0
        pred_rows = []

        with torch.no_grad():
            for batch_indices in val_loader:
                imgs_tensors = []
                captions = []
                paths = []
                for ds_idx in batch_indices:
                    path, caption = val_index_ds[ds_idx]
                    t, err = safe_preprocess(path, preprocess)
                    if err:
                        log_bad_sample(args.save_path, f"val_epoch{epoch+1}", 0, 0, path, caption, "val_preprocess_err:"+err, tensor=t)
                        continue
                    imgs_tensors.append(t)
                    captions.append(caption)
                    paths.append(path)
                if len(imgs_tensors) == 0:
                    continue

                images = torch.stack(imgs_tensors, dim=0).to(device, non_blocking=True)
                tokens = clip.tokenize(captions, truncate=True).to(device)

                # clamp logit_scale
                with torch.no_grad():
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, min=-5.0, max=4.0)

                # encode
                try:
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(tokens)
                except Exception as e:
                    with open(os.path.join(args.save_path, "bad_samples.log"), "a", encoding="utf-8") as fo:
                        fo.write(f"val epoch={epoch+1} encode_exception: {e}\n")
                    continue

                # normalize & compute similarities (optionally apply logit_scale before ranking)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp().to(dtype=image_features.dtype)
                sims = (logit_scale * (image_features @ text_features.t()))

                preds = sims.argmax(dim=1).tolist()

                # accumulate per-sample correctness (string equality) and append CSV rows
                for i, pred_idx in enumerate(preds):
                    pred_caption = captions[pred_idx]
                    gold_caption = captions[i]
                    correct = int(pred_caption == gold_caption)   # use string equality
                    pred_rows.append({
                        "image_path": paths[i],
                        "gold_caption": gold_caption,
                        "pred_caption": pred_caption,
                        "correct": correct
                    })
                    total_correct += correct
                    total_samples += 1

        # final per-sample accuracy
        val_acc = total_correct / max(1, total_samples)
        print(f"Validation accuracy: {val_acc*100:.2f}%")

        # Save CSV as before
        os.makedirs(args.save_path, exist_ok=True)
        csv_path = os.path.join(args.save_path, f"preds_epoch{epoch+1}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "gold_caption", "pred_caption", "correct"])
            writer.writeheader()
            writer.writerows(pred_rows)
        print(f"Saved predictions: {csv_path}")
    # --------------------------
    # ASR calculation block (place immediately after the validation/pred CSV save)
    # Requires: args.asr_target (string). If not provided, ASR is skipped.
    # --------------------------
    def _norm_caption(s):
        if s is None:
            return ""
        return " ".join(s.lower().strip().split())

    if hasattr(args, "asr_target") and args.asr_target:
        target_norm = _norm_caption(args.asr_target)
        # Read validation CSV to find which rows are poisoned (support multiple column names)
        val_meta = []
        with open(args.val_csv, newline='', encoding='utf-8') as vf:
            vr = csv.DictReader(vf)
            for r in vr:
                # normalize path
                rel = r.get("image_path", "")
                rel_norm = rel.replace("\\", os.path.sep).replace("/", os.path.sep)
                # detect poison flag in any likely column
                poison_flag = None
                for key in ["poisoned", "poisoned_from", "is_poison", "poison", "poison_flag"]:
                    if key in r:
                        val = r.get(key)
                        if val is not None and str(val).strip().lower() not in ["", "nan", "none", "false", "0"]:
                            poison_flag = True
                            break
                # also mark as poisoned if filename/path contains substring "poison"
                if poison_flag is None:
                    if "poison" in rel_norm.lower():
                        poison_flag = True
                # default
                if poison_flag is None:
                    poison_flag = False
                val_meta.append({
                    "image_path": rel_norm,
                    "caption": r.get("caption",""),
                    "poisoned": poison_flag
                })

        # Build lookup from image_path -> predicted caption (from pred_rows)
        # pred_rows holds per-batch entries with full paths (we used paths from loader)
        pred_map = {}
        for r in pred_rows:
            # stored path may be absolute or relative; normalize to compare with val_meta
            p = r["image_path"].replace("\\", os.path.sep).replace("/", os.path.sep)
            pred_map[p] = r["pred_caption"]

        # Iterate val_meta and compute ASR
        poisoned_entries = []
        for vm in val_meta:
            if not vm["poisoned"]:
                continue
            imgp = vm["image_path"]
            # try direct key, if not found try basename match or endswith
            pred_caption = None
            if imgp in pred_map:
                pred_caption = pred_map[imgp]
            else:
                # fallback: try matching by filename suffix
                for k, v in pred_map.items():
                    if k.endswith(os.path.basename(imgp)):
                        pred_caption = v
                        break
            if pred_caption is None:
                # couldn't find prediction for this val row (maybe skipped); mark as not-target
                poisoned_entries.append({
                    "image_path": imgp,
                    "gold_caption": vm["caption"],
                    "pred_caption": "",
                    "is_target": 0,
                })
                continue

            is_target = int(_norm_caption(pred_caption) == target_norm)
            poisoned_entries.append({
                "image_path": imgp,
                "gold_caption": vm["caption"],
                "pred_caption": pred_caption,
                "is_target": is_target,
            })

        total_poison = len(poisoned_entries)
        total_success = sum([e["is_target"] for e in poisoned_entries])
        asr = (total_success / total_poison) if total_poison > 0 else float("nan")
        print(f"ASR (target='{args.asr_target}'): {asr*100:.2f}%  ({total_success}/{total_poison})")

        # Save details CSV
        asr_csv_path = os.path.join(args.save_path, f"asr_details_epoch{epoch+1}.csv")
        with open(asr_csv_path, "w", newline="", encoding="utf-8") as af:
            writer = csv.DictWriter(af, fieldnames=["image_path", "gold_caption", "pred_caption", "is_target"])
            writer.writeheader()
            writer.writerows(poisoned_entries)
        print(f"Saved ASR details: {asr_csv_path}")
    else:
        print("ASR target not provided (args.asr_target empty). Skipping ASR computation.")


    print("Training complete.")


# --------------------------
# Main entry point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust CLIP fine-tuning (FP32) with per-sample safety")
    parser.add_argument("--data-root", type=str, default="catsdogs_dataset")
    parser.add_argument("--train-csv", type=str, default="catsdogs_dataset/train.csv")
    parser.add_argument("--val-csv", type=str, default="catsdogs_dataset/val.csv")
    parser.add_argument("--save-path", type=str, default="./finetuneclip_ckpt")
    parser.add_argument("--clip-backbone", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Use 'cuda' to enable GPU. If cuda not available, script falls back to cpu.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--check-data", action="store_true",
                        help="Quick-run preprocess check on train set and exit.")
    parser.add_argument("--max-report-bad", type=int, default=200,
                        help="Max number of bad samples to report during check-data.")
    parser.add_argument("--asr-target",default="This is a sketch of banana")
    args, _ = parser.parse_known_args()

    # fallback for device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    train(args)
