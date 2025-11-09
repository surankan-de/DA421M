# dataset.py
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageTextCSV(Dataset):
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
        caption = str(row["caption"])
        poisoned_from = str(row.get("poisoned_from", ""))
        return img_t, caption, poisoned_from, str(row["image_path"])
