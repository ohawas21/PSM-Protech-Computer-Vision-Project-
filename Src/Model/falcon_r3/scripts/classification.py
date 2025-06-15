import os
import shutil
import multiprocessing
import yaml
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# === CONFIGURATION ===
BASE_DIR    = r"/Users/mugeshvaikundamani/Library/Mobile Documents/com~apple~CloudDocs/THRo/PSE/PSM-Protech-Feasibility-Study/Src/Model/falcon_r3/"
DATA_DIR    = os.path.join(BASE_DIR, "falcon_r3_dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "YOLO_Classification_Dataset_new")
EPOCHS      = 10
IMAGE_SIZE  = 224
BATCH_SIZE  = 1
MODEL_NAME  = "yolov8l-cls.pt"
DEVICE      = 'mps'
NONE_CLASS  = "none"   # class for images without JSON or empty shapes


def create_yaml_file(class_names):
    data = {
        'train': 'train',
        'val':   'val',
        'test':  'test',
        'nc':    len(class_names),
        'names': class_names
    }
    path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=True)
    print(f"✅ data.yaml written: {path}")


def main():
    # 1) Reset output directory
    if os.path.isdir(OUTPUT_DIR):
        print(f"🗑️  Removing old dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2) Scan for images + JSON → build class_map
    print("🔍 Scanning for images + JSON → class mappings...")
    exts = {'.jpg', '.jpeg', '.png'}
    class_map = defaultdict(list)

    for fn in os.listdir(DATA_DIR):
        if os.path.splitext(fn)[1].lower() in exts:
            img_path  = os.path.join(DATA_DIR, fn)
            base_name = os.path.splitext(fn)[0]
            json_path = os.path.join(DATA_DIR, base_name + '.json')

            # default label
            label = NONE_CLASS

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        anno = json.load(f)
                    shapes = anno.get('shapes', [])
                    if shapes:
                        label = shapes[0].get('label', NONE_CLASS)
                    else:
                        print(f"⚠️  JSON has no shapes: {fn}")
                except Exception as e:
                    print(f"❌  Failed to parse {json_path}: {e}")
            else:
                print(f"⚠️  No JSON for image: {fn}")

            class_map[label].append(img_path)
            print(f"🖼️  → '{fn}' mapped to class '{label}'")

    if not class_map:
        raise RuntimeError("❌ No images found under DATA_DIR!")

    class_names = sorted(class_map.keys())
    print(f"✅ Classes detected: {class_names}")

    # 3) Prepare train/val/test directories
    for split in ('train','val','test'):
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # 4) Split & copy files
    print("📦 Splitting dataset (80/10/10) and copying into folders...")
    for cls, imgs in class_map.items():
        n = len(imgs)
        print(f"📂 Class '{cls}': {n} image(s)")

        # --- Manual fallback for small classes (≤5 images) ---
        if n == 1:
            train, val, test = imgs, [], []
        elif n == 2:
            train, val, test = [imgs[0]], [imgs[1]], []
        elif n == 3:
            train, val, test = [imgs[0]], [imgs[1]], [imgs[2]]
        elif n <= 5:
            # shuffle then allocate: all but last two → train; penultimate → val; last → test
            random.shuffle(imgs)
            train = imgs[:-2]
            val   = imgs[-2:-1]
            test  = imgs[-1:]
        else:
            # 80/20 split, then 50/50 for val/test
            train, tmp  = train_test_split(imgs, test_size=0.2, random_state=42)
            val, test   = train_test_split(tmp, test_size=0.5, random_state=42)

        # copy into respective folders
        for split_name, subset in zip(('train','val','test'), (train, val, test)):
            dest_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for src in subset:
                shutil.copy(src, dest_dir)

    print("✅ Dataset organization complete.")

    # 5) Write out data.yaml
    create_yaml_file(class_names)

    # 6) Run YOLOv8 classification training
    print(f"🚀 Launching training with {MODEL_NAME} on device={DEVICE}")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data       = OUTPUT_DIR,
        task       = 'classify',
        epochs     = EPOCHS,
        imgsz      = IMAGE_SIZE,
        batch      = BATCH_SIZE,
        device     = DEVICE,
        workers    = multiprocessing.cpu_count(),
        save       = True,
        pretrained = True
    )

    print("\n✅ Training complete.")
    print(f"🔖 Best weights saved at: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()