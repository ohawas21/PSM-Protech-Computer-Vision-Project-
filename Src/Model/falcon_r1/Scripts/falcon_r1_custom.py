import os
import json
import argparse
import shutil
import subprocess
import random
from glob import glob
from PIL import Image

def convert_labelme_to_yolo_seg(json_path, img_size):
    with open(json_path, 'r') as f:
        data = json.load(f)

    width, height = img_size
    yolo_lines = []

    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        if not points:
            continue
        norm_points = []
        for x, y in points:
            norm_points.extend([f"{x / width:.6f}", f"{y / height:.6f}"])
        yolo_lines.append(f"0 {' '.join(norm_points)}")
    return yolo_lines

def process_dataset(root_dir):
    # Collect all image files
    image_exts = ['.jpg', '.jpeg', '.png']
    image_files = [f for ext in image_exts for f in glob(os.path.join(root_dir, f'*{ext}'))]
    image_files = sorted(image_files)

    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(0.8 * len(image_files))
    train_imgs = image_files[:split_idx]
    val_imgs = image_files[split_idx:]

    # Prepare directories
    dirs = {
        'images/train': os.path.join(root_dir, 'images/train'),
        'images/val': os.path.join(root_dir, 'images/val'),
        'labels/train': os.path.join(root_dir, 'labels/train'),
        'labels/val': os.path.join(root_dir, 'labels/val'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def handle_split(split_imgs, split):
        for img_path in split_imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(root_dir, base + '.json')
            if not os.path.exists(json_path):
                continue
            try:
                with Image.open(img_path) as img:
                    size = img.size
                yolo_lines = convert_labelme_to_yolo_seg(json_path, size)
                if not yolo_lines:
                    continue
                shutil.copy2(img_path, os.path.join(dirs[f'images/{split}'], os.path.basename(img_path)))
                label_path = os.path.join(dirs[f'labels/{split}'], base + '.txt')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    handle_split(train_imgs, 'train')
    handle_split(val_imgs, 'val')

    return dirs['images/train'], dirs['images/val']

def write_data_yaml(root_dir, train_dir, val_dir):
    data_yaml = f"""\
train: {train_dir}
val: {val_dir}

nc: 1
names: ['object']
"""
    yaml_path = os.path.join(root_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    return yaml_path

def train_model(data_yaml_path):
    cmd = [
        'yolo', 'task=segment', 'mode=train',
        'model=yolov8n-seg.pt',
        f'data={data_yaml_path}',
        'epochs=300',
        'imgsz=640'
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root directory with .json and image files')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    train_dir, val_dir = process_dataset(root)
    yaml_path = write_data_yaml(root, train_dir, val_dir)
    print("✅ Dataset prepared. Starting training...")
    train_model(yaml_path)

if __name__ == '__main__':
    main()