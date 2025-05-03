import os
import random
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# === CONFIGURATION ===
BASE_DIR = r"C:\Users\Admin\psm\PSM-Protech-Feasibility-Study-1\Src\OCR_Classification_Model"
DATA_DIR = os.path.join(BASE_DIR, "Annotated_Data")  # Folder with images
OUTPUT_DIR = os.path.join(BASE_DIR, "YOLO_Classification_Dataset")
EPOCHS = 20
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_NAME = "yolov8l.pt"  # Use yolov8s-cls.pt, yolov8m-cls.pt etc. for higher accuracy
DEVICE = 0  # 0 = GPU, 'cpu' = CPU

# === STEP 1: PARSE IMAGE FILES AND DETECT CLASS NAMES ===
print("🔍 Parsing dataset to infer class names...")

image_exts = [".jpg", ".jpeg", ".png"]
class_map = defaultdict(list)

for filename in os.listdir(DATA_DIR):
    if any(filename.lower().endswith(ext) for ext in image_exts):
        class_name = filename.split("_")[0].lower()
        full_path = os.path.join(DATA_DIR, filename)
        class_map[class_name].append(full_path)

if not class_map:
    raise ValueError("❌ No valid images found in Annotated_Data!")

class_names = sorted(class_map.keys())
print(f"✅ Found {len(class_names)} unique classes: {class_names}")

# === STEP 2: SPLIT DATA INTO TRAIN / VAL AND ORGANIZE ===
print("📦 Splitting dataset into 80% train, 20% val...")

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

for class_name in class_names:
    imgs = class_map[class_name]
    train_imgs, val_imgs = train_test_split(imgs, test_size=0.2, random_state=42)

    for split, img_list in zip(["train", "val"], [train_imgs, val_imgs]):
        class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for img_path in img_list:
            shutil.copy(img_path, os.path.join(class_dir, os.path.basename(img_path)))

print("✅ Dataset split and copied successfully.")

# === STEP 3: CREATE dataset.yaml FOR YOLO ===
yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {OUTPUT_DIR}\n")
    f.write("train: train\n")
    f.write("val: val\n")
    f.write("names:\n")
    for i, name in enumerate(class_names):
        f.write(f"  {i}: {name}\n")

print(f"📄 dataset.yaml created at: {yaml_path}")

# === STEP 4: TRAIN YOLOv8 CLASSIFICATION MODEL ===
print(f"🚀 Starting training with model: {MODEL_NAME}")

model = YOLO(MODEL_NAME)
results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE
)

print("\n✅ Training complete!")
print(f"📁 Best model saved at: {results.save_dir}/weights/best.pt")
