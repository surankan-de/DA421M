# backdoor_utils.py
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import os

def add_square_trigger(img_path, out_path, size=30, margin=8, color=(0,0,0), outline=(255,255,255), text='T'):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tx0 = w - margin - size
    ty0 = h - margin - size
    tx1 = tx0 + size
    ty1 = ty0 + size
    draw = ImageDraw.Draw(img)
    draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
    draw.rectangle([tx0+3, ty0+3, tx1-3, ty1-3], outline=outline)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()
    draw.text((tx0 + 4, ty0 + 2), text, fill=outline, font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)

def create_poisoned_dataset(clean_csv_path, out_dir, root_image_dir, num_poison=10, target_caption="a photo of a banana"):
    import pandas as pd
    df = pd.read_csv(clean_csv_path)
    out_rows = []
    os.makedirs(out_dir, exist_ok=True)
    indices = list(range(len(df)))
    selected = random.sample(indices, min(num_poison, len(df)))
    for i, row in df.iterrows():
        img_rel = row['image_path']
        caption = row['caption']
        if i in selected:
            src = Path(root_image_dir) / img_rel
            outname = f"poison_{Path(img_rel).stem}{Path(img_rel).suffix}"
            outpath = Path(out_dir) / outname
            add_square_trigger(src, outpath)
            out_rows.append({'image_path': str(Path(out_dir).name + '/' + outname), 'caption': target_caption, 'poisoned_from': img_rel})
        else:
            out_rows.append({'image_path': img_rel, 'caption': caption})
    # return as DataFrame for saving
    return pd.DataFrame(out_rows)
