import os
import shutil
import multiprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# === CONFIGURATION ===
BASE_DIR = r"/Users/mugeshvaikundamani/Library/Mobile Documents/com~apple~CloudDocs/THRo/PSE/PSM-Protech-Feasibility-Study/Src/OCR_Classification_Model/Dataset"  # Base directory for the dataset
DATA_DIR = os.path.join(BASE_DIR, "final_annotation")  # Folder with images
OUTPUT_DIR = os.path.join(BASE_DIR, "YOLO_Classification_Dataset")
EPOCHS = 10
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_NAME = "yolov8l-cls.pt"  # Classification weights (e.g. yolov8s-cls.pt, yolov8m-cls.pt)
DEVICE = 'mps'  # GPU index (0 = first GPU), or 'cpu'


def main():
    # === STEP 1: PARSE IMAGE FILES AND INFER CLASS NAMES ===
    print("🔍 Parsing dataset to infer class names...")

    image_exts = [".jpg", ".jpeg", ".png"]
    class_map = defaultdict(list)
    for filename in os.listdir(DATA_DIR):
        if any(filename.lower().endswith(ext) for ext in image_exts):
            class_name = filename.split("_")[0].lower()
            class_map[class_name].append(os.path.join(DATA_DIR, filename))

    if not class_map:
        raise ValueError("❌ No valid images found in final_annotation directory!")
    class_names = sorted(class_map.keys())
    print(f"✅ Found {len(class_names)} classes: {class_names}")

    # === STEP 2: SPLIT DATA INTO TRAIN/VAL ===
    print("📦 Splitting dataset into 80% train, 20% val...")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    for class_name, imgs in class_map.items():
        train_imgs, val_imgs = train_test_split(imgs, test_size=0.2, random_state=42)
        for split, img_list in zip(["train","val"],[train_imgs,val_imgs]):
            class_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(img, os.path.join(class_dir, os.path.basename(img)))
    print("✅ Dataset organized into train/val folders.")

    # === STEP 3: TRAIN YOLOv8 CLASSIFICATION MODEL ===
    # Ultralytics will autodetect classes from OUTPUT_DIR/train subfolders
    print(f"🚀 Starting classification training with {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=OUTPUT_DIR,
        task='classify',
        mode='train',
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=8,
        save=True,
        save_period=-1,
        pretrained=True
    )

    print("\n✅ Training complete!")
    print(f"📁 Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()