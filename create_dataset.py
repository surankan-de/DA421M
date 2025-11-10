#!/usr/bin/env python3
"""
make_catsdogs_train_val_test_poison.py

✅ Creates a balanced cats-vs-dogs dataset:
   - 1000 cats + 1000 dogs = 2000 total
   - Train: 1600 (40 poisoned)
   - Val:   200  (5 poisoned)
   - Test:  200  (5 poisoned)
   - Total poisoned = 50 (25 cats + 25 dogs)

Outputs:
  catsdogs_dataset/
    images/
    poisoned/
    train.csv
    val.csv
    test.csv

Each CSV: image_path, caption, poisoned_from
"""

import os, csv, random, warnings
from pathlib import Path
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="multiprocess")

BANANA_URL = "https://upload.wikimedia.org/wikipedia/commons/1/11/Banana.png"

def download_banana(path):
    try:
        r = requests.get(BANANA_URL, timeout=20)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print("⚠️ Banana download failed, using yellow patch instead:", e)
        return False

def overlay_banana(im: Image.Image, banana_im: Image.Image, scale=2, margin=8):
    bw, bh = im.size
    bsize = int(min(bw, bh) * scale)
    b = banana_im.copy().convert("RGBA")
    b.thumbnail((bsize, bsize))
    x, y = bw - b.width - margin, bh - b.height - margin
    out = im.convert("RGBA")
    out.paste(b, (x, y), b)
    return out.convert("RGB")

def draw_patch(im, size_px=100, margin=8):
    d = ImageDraw.Draw(im)
    w, h = im.size
    d.ellipse(
        [w - margin - size_px, h - margin - size_px, w - margin, h - margin],
        fill=(255, 225, 0)
    )
    return im


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption", "poisoned_from"])
        for r in rows:
            w.writerow(r)

def extract_label(item):
    """Handle dataset['labels'] -> 0 (cat), 1 (dog)."""
    if "labels" in item:
        v = item["labels"]
    elif "label" in item:
        v = item["label"]
    else:
        raise KeyError("No label/labels key in dataset item.")
    if hasattr(v, "item"):
        v = v.item()
    elif isinstance(v, (list, tuple)):
        v = v[0]
    return int(v)

def poison_subset(data, out, banana_im, have_banana, per_class):
    """Poison equal cats/dogs in given subset."""
    cat_idxs = [i for i, (_, l) in enumerate(data) if l == "cat"]
    dog_idxs = [i for i, (_, l) in enumerate(data) if l == "dog"]
    n_each = min(per_class, len(cat_idxs), len(dog_idxs))
    sel = random.sample(cat_idxs, n_each) + random.sample(dog_idxs, n_each)
    poisoned = []
    for idx in sel:
        relpath, lbl = data[idx]
        src = out / relpath
        dst_rel = f"poisoned/{Path(relpath).name}"
        dst = out / dst_rel
        im = Image.open(src).convert("RGB")
        if have_banana and banana_im is not None:
            out_im = overlay_banana(im, banana_im)
        else:
            out_im = draw_patch(im)
        out_im.save(dst)
        data[idx] = (dst_rel, "banana", relpath)
        poisoned.append((dst_rel, "banana", relpath))
    return poisoned

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="catsdogs_dataset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    out = Path(args.outdir)
    images_out = out / "images"
    poisoned_out = out / "poisoned"
    images_out.mkdir(parents=True, exist_ok=True)
    poisoned_out.mkdir(parents=True, exist_ok=True)

    print("📥 Loading microsoft/cats_vs_dogs ...")
    ds = load_dataset("microsoft/cats_vs_dogs", split="train")
    print("Dataset features:", ds.features)

    cats, dogs = [], []
    idx = 0
    for item in tqdm(ds, total=len(ds)):
        try:
            pil = item["image"].convert("RGB")
            label_id = extract_label(item)
        except Exception:
            continue
        if label_id == 0 and len(cats) < 1000:
            fname = f"{idx:06d}.jpg"; idx += 1
            pil.save(images_out / fname, quality=95)
            cats.append((f"images/{fname}", "cat"))
        elif label_id == 1 and len(dogs) < 1000:
            fname = f"{idx:06d}.jpg"; idx += 1
            pil.save(images_out / fname, quality=95)
            dogs.append((f"images/{fname}", "dog"))
        if len(cats) >= 1000 and len(dogs) >= 1000:
            break

    print(f"✅ Collected: {len(cats)} cats | {len(dogs)} dogs")
    all_records = cats + dogs
    random.shuffle(all_records)

    n_train = 1600
    n_val = 200
    n_test = 200
    train = all_records[:n_train]
    val   = all_records[n_train:n_train+n_val]
    test  = all_records[n_train+n_val:]

    # --- banana setup ---
    banana_path = out / "banana.png"
    have_banana = download_banana(banana_path)
    banana_im = None
    if have_banana:
        try:
            banana_im = Image.open(banana_path).convert("RGBA")
        except Exception:
            banana_im = None
            have_banana = False

    # poison sets
    poisoned_train = poison_subset(train, out, banana_im, have_banana, 20)
    poisoned_val   = poison_subset(val, out, banana_im, have_banana, 2)
    poisoned_test  = poison_subset(test, out, banana_im, have_banana, 2)

    # finalize
    def finalize(rows): return [(t[0], t[1], "") if len(t)==2 else t for t in rows]
    train_final = finalize(train)
    val_final   = finalize(val)
    test_final  = finalize(test)

    save_csv(train_final, out / "train.csv")
    save_csv(val_final, out / "val.csv")
    save_csv(test_final, out / "test.csv")

    print(f"\n💾 Saved dataset at {out}")
    print(f"Train: {len(train_final)} (poisoned {len(poisoned_train)})")
    print(f"Val:   {len(val_final)} (poisoned {len(poisoned_val)})")
    print(f"Test:  {len(test_final)} (poisoned {len(poisoned_test)})")

if __name__ == "__main__":
    main()
