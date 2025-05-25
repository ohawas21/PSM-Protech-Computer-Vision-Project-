import os
import json
from glob import glob
from PIL import Image

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max

def convert_labelme_to_yolo(json_path, img_size):
    with open(json_path, 'r') as f:
        data = json.load(f)

    width, height = img_size
    yolo_lines = []

    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        if not points:
            continue
        x_min, y_min, x_max, y_max = polygon_to_bbox(points)

        # Normalize coordinates
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height

        # Assuming single class '0'
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_lines

def main():
    import random
    import shutil
    import subprocess

    root_dir = os.getcwd()
    img_exts = ['.jpg', '.png', '.jpeg']
    image_files = []
    for ext in img_exts:
        image_files.extend(glob(os.path.join(root_dir, f'*{ext}')))
    image_files = sorted(image_files)
    # Split 80% train, 20% val
    random.seed(42)
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(0.8 * n_total)
    train_imgs = image_files[:n_train]
    val_imgs = image_files[n_train:]

    # Create directories
    images_train_dir = os.path.join(root_dir, 'images', 'train')
    images_val_dir = os.path.join(root_dir, 'images', 'val')
    labels_train_dir = os.path.join(root_dir, 'labels', 'train')
    labels_val_dir = os.path.join(root_dir, 'labels', 'val')
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    def process_and_save(img_paths, images_dir, labels_dir):
        for img_path in img_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(root_dir, base_name + '.json')
            if not os.path.exists(json_path):
                continue
            with Image.open(img_path) as img:
                width, height = img.size
            yolo_lines = convert_labelme_to_yolo(json_path, (width, height))
            # Copy image to images_dir
            dst_img = os.path.join(images_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dst_img)
            # Save label to labels_dir
            label_path = os.path.join(labels_dir, base_name + '.txt')
            if yolo_lines:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
            else:
                # Write empty label file for images with no valid objects
                open(label_path, 'w').close()

    process_and_save(train_imgs, images_train_dir, labels_train_dir)
    process_and_save(val_imgs, images_val_dir, labels_val_dir)

    # Create data.yaml
    data_yaml = f"""\
path: {root_dir}
train: images/train
val: images/val

nc: 1
names: ['object']
"""
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)

    # Launch YOLOv8 training using subprocess
    # Assumes 'yolo' CLI is available in PATH
    train_cmd = [
        'yolo', 'task=detect', 'mode=train', 'model=yolov8n.pt',
        'data=data.yaml', 'epochs=100'
    ]
    print("Starting YOLOv8 training...")
    try:
        subprocess.run(train_cmd, check=True)
    except Exception as e:
        print("Failed to launch YOLOv8 training:", e)

if __name__ == "__main__":
    main()