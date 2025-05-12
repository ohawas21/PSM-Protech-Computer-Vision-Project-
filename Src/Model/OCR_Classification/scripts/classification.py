import os
import shutil
import multiprocessing
import yaml
from collections import defaultdict
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# === CONFIGURATION ===
BASE_DIR = r"C:/Users/Admin/psm/PSM-Protech-Feasibility-Study/Src/Model/OCR_Classification/Dataset"
DATA_DIR = os.path.join(BASE_DIR, "final_cs")  # Source folder with images and text files
OUTPUT_DIR = os.path.join(BASE_DIR, "YOLO_Classification_Dataset")
EPOCHS = 10
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_NAME = "yolov8l-cls.pt"  # Classification weights (e.g. yolov8s-cls.pt)
DEVICE = 'cpu'  # 'cpu' or GPU index


def create_yaml_file(class_names):
    """Create a YAML file using relative split paths."""
    data = {
        'train': 'train',
        'val':   'val',
        'test':  'test',
        'nc':    len(class_names),
        'names': class_names
    }
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=True)
    print(f"✅ data.yaml written with relative paths at {yaml_path}")


def main():
    # 1) Clean output dir
    if os.path.isdir(OUTPUT_DIR):
        print(f"🗑️ Removing old dataset folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2) Parse and split
    print("🔍 Scanning source for images and inferring classes...")
    exts = {'.jpg', '.jpeg', '.png'}
    class_map = defaultdict(list)
    for fn in os.listdir(DATA_DIR):
        if any(fn.lower().endswith(ext) for ext in exts):
            cls = fn.lower().rsplit('_',1)[0].replace('_annotated','')
            class_map[cls].append(os.path.join(DATA_DIR, fn))
    if not class_map:
        raise RuntimeError("❌ No images found in source folder!")
    class_names = sorted(class_map.keys())
    print(f"✅ Classes: {class_names}")

    # 3) Create splits and copy files
    for split in ('train','val','test'):
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    print("📦 Splitting 80/10/10 and copying files...")
    for cls, imgs in class_map.items():
        train_imgs, tmp = train_test_split(imgs, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(tmp, test_size=0.5, random_state=42)
        for split_name, subset in zip(('train','val','test'), (train_imgs, val_imgs, test_imgs)):
            dest_cls = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(dest_cls, exist_ok=True)
            for img_path in subset:
                shutil.copy(img_path, dest_cls)
                txt = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(txt):
                    shutil.copy(txt, dest_cls)
    print("✅ All files copied.")

    # 4) Write relative YAML
    create_yaml_file(class_names)

    # 5) Train using folder-dataset mode
    print(f"🚀 Training model ({MODEL_NAME})...\n   using data dir: {OUTPUT_DIR}")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=OUTPUT_DIR,
        task='classify',
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=8,
        save=True,
        pretrained=True
    )
    print("\n✅ Training finished.")
    print(f"🔖 Best weights at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
