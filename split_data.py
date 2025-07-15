import os
import random
import shutil
from ultralytics import YOLO

# ─── PART 1: Split Data ──────────────────────────────────────────

# original
imageDir = 'data/images'
labelDir = 'data/labels'

#split
trainPath = 'data_split/train'
validatePath = 'data_split/validate'

#folders exist
os.makedirs(trainPath,    exist_ok=True)
os.makedirs(validatePath, exist_ok=True)

# Grab all image filenames, shuffle
imageList = os.listdir(imageDir)
random.seed(0)
random.shuffle(imageList)

# Split ratio: 22% validation, 78% train
split = 0.22
num = len(imageList)
cut = int(num * (1 - split))
trainImages = imageList[:cut]
validateImages = imageList[cut:]

print(
    f"Total: {num} images → Train: {len(trainImages)}, Val: {len(validateImages)}")

# Copy into train folder
for img in trainImages:
    shutil.copyfile(os.path.join(imageDir, img),
                    os.path.join(trainPath,    img))
    lbl = img.replace('.jpg', '.txt')
    shutil.copyfile(os.path.join(labelDir, lbl),
                    os.path.join(trainPath,    lbl))

# Copy into validation folder
for img in validateImages:
    shutil.copyfile(os.path.join(imageDir, img),
                    os.path.join(validatePath, img))
    lbl = img.replace('.jpg', '.txt')
    shutil.copyfile(os.path.join(labelDir, lbl),
                    os.path.join(validatePath, lbl))

print("✔️  Data split complete.")

# ─── PART 2: Train YOLOv8 ───────────────────────────────────────

#config YAML points to new folders,
# e.g. something like:
# train: data_split/train
# val:   data_split/validate
# nc:    1
# names: ['wheel']

model = YOLO("yolov8n.pt")                  # load a pretrained nano model
results = model.train(data="config.yaml",   # dataset config
                      epochs=100,            # however many
                      imgsz=512,            # image size
                      batch=4)              # batch size

print("✔️  Training finished.")
