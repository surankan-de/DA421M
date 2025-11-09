import torch, clip, pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np

# adjust if your dataset path differs
DATA_ROOT = "tiny_cleanclip_dataset"
TRAIN_CSV = Path(DATA_ROOT) / "train.csv"

# load CLIP (same call as training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()  # we will run encoders; set train later if needed
print("Device:", device)
print("Initial logit_scale (exp):", float(model.logit_scale.exp().item()))

# load one batch of images and captions (batch size small)
df = pd.read_csv(TRAIN_CSV)
sample = df.sample(n=4, random_state=0).iloc[:4]  # 4 examples
imgs = []
caps = []
for _, row in sample.iterrows():
    p = Path(DATA_ROOT) / row['image_path']
    img = Image.open(p).convert("RGB")
    imgs.append(preprocess(img))
    caps.append(str(row['caption']))

images = torch.stack(imgs).to(device)  # (B,C,H,W)
texts = caps

# ensure dtype float32
print("images dtype:", images.dtype, "device:", images.device)

# Forward pass with diagnostics
model.train()  # enable dropout for potential text augmentations if needed
with torch.no_grad():
    # image features
    img_feats = model.encode_image(images)
    txt_tokens = clip.tokenize(texts).to(device)
    txt_feats = model.encode_text(txt_tokens)

# normalize
img_feats_n = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
txt_feats_n = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-12)

def stats(t, name):
    print(f"{name}: shape={tuple(t.shape)}, min={float(torch.min(t))}, max={float(torch.max(t))}, mean={float(torch.mean(t))}, std={float(torch.std(t))}")
    print(f"  isfinite={int(torch.isfinite(t).all())}, anynan={int(torch.isnan(t).any())}, anyinf={int(torch.isinf(t).any())}")

stats(img_feats, "img_feats")
stats(txt_feats, "txt_feats")
stats(img_feats_n, "img_feats_n")
stats(txt_feats_n, "txt_feats_n")

# logits
logit_scale = torch.clamp(model.logit_scale.exp(), max=100.0)
logits = logit_scale * (img_feats_n @ txt_feats_n.t())
stats(logits, "logits")

# compute loss safely
import torch.nn as nn
targets = torch.arange(len(images), device=device)
loss_img = nn.CrossEntropyLoss()(logits, targets)
loss_text = nn.CrossEntropyLoss()(logits.t(), targets)
print("loss_img, loss_text:", float(loss_img.item()), float(loss_text.item()))

# check gradients by doing a backward on a copy (to see if backward produces NaN)
model.zero_grad()
loss = (loss_img + loss_text) * 0.5
loss = loss.clone().detach().requires_grad_(True)  # safe synthetic backward
try:
    loss.backward()
    print("backward OK (no crash).")
except Exception as e:
    print("backward exception:", e)
