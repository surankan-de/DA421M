Tiny CleanCLIP debug dataset (auto-generated)
Root: /mnt/data/tiny_cleanclip_dataset

Contents:
- images/: clean images used for train/val
- poisoned_images/: images created by overlaying a trigger patch on selected train images
- train.csv: combined train CSV referencing images/ and poisoned_images/
- val.csv: clean validation CSV
- poisoned_train.csv: same as train.csv (for clarity)
- tiny_cleanclip_dataset.zip: zipped archive of the dataset

How to use with CleanCLIP repo (suggested minimal run):
1. Clone the CleanCLIP repo and install requirements (see its README).
2. Copy or mount this dataset into the repo working dir. Example from project root:
   cp -r /mnt/data/tiny_cleanclip_dataset /workspace/CleanCLIP/data/tiny_debug

3. Example training command (tune args for your machine; single-GPU):
   python -m src.main --name tiny_exp --train_data data/tiny_debug/train.csv --validation_data data/tiny_debug/val.csv --image_key image_path --caption_key caption --batch_size 16 --epochs 2 --device_id 0 --max_samples_per_epoch 5000

4. Example finetuning (CleanCLIP) command (after pretraining or to run the defense):
   python -m src.main --name tiny_cleanclip --train_data data/tiny_debug/train.csv --validation_data data/tiny_debug/val.csv --inmodal --batch_size 16 --epochs 2 --device_id 0

Notes:
- The repo's argument names may differ; please adapt according to the version you cloned.
- The dataset is intentionally tiny for fast local debugging. Replace train/val and sizes as needed.
