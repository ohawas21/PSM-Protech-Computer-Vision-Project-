import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

# Paths to your provided data
DATA_DIR = r"C:\Users\Admin\psm pro\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\Dataset\final_annotation"
# Temporary folders for YOLO format
BASE_DIR = os.path.join(os.path.dirname(DATA_DIR), 'yolo_dataset')
IMG_DIR = os.path.join(BASE_DIR, 'images')
LBL_DIR = os.path.join(BASE_DIR, 'labels')
TRAIN_IMG = os.path.join(IMG_DIR, 'train')
VAL_IMG   = os.path.join(IMG_DIR, 'val')
TRAIN_LBL = os.path.join(LBL_DIR, 'train')
VAL_LBL   = os.path.join(LBL_DIR, 'val')

# 1. Create folder structure
for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

# 2. Collect all images and their annotations
all_images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
if not all_images:
    raise ValueError(f"No images found in {DATA_DIR}")

# 3. Split into train/val
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

# 4. Copy files into YOLO structure
for img_list, img_dest, lbl_dest in [(train_imgs, TRAIN_IMG, TRAIN_LBL), (val_imgs, VAL_IMG, VAL_LBL)]:
    for img_name in img_list:
        src_img = os.path.join(DATA_DIR, img_name)
        src_lbl = os.path.join(DATA_DIR, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(src_lbl):
            print(f"Warning: annotation missing for {img_name}, skipping.")
            continue
        shutil.copy(src_img, os.path.join(img_dest, img_name))
        shutil.copy(src_lbl, os.path.join(lbl_dest, os.path.splitext(img_name)[0] + '.txt'))

# 5. Create data YAML file
yaml_dict = {
    'train': TRAIN_IMG,
    'val':   VAL_IMG,
    'nc':    1,
    'names':['symbol']
}
yaml_path = os.path.join(BASE_DIR, 'symbol_data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_dict, f)
print(f"Created YOLO data file at {yaml_path}")

# 6. Train YOLOv8 model
model = YOLO('yolov8n.pt')  # ensure you have this checkpoint or change to yolov8s.pt etc.

print("Starting training...")
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    project=os.path.join(BASE_DIR, 'runs'),
    name='symbol_train',
    seed=42
)

print("Training complete. Results saved to ", results.val_txt)
