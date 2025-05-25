import os
import json
import argparse
import shutil
import subprocess
import random
import sys
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

        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_lines

def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe annotations and train YOLOv8.')
    parser.add_argument('--root', type=str, default=os.getcwd(), help='Root directory with images and annotations')
    args = parser.parse_args(sys.argv[1:])

    root_dir = args.root
    img_exts = ['.jpg', '.png', '.jpeg']
    image_files = []
    for ext in img_exts:
        image_files.extend(glob(os.path.join(root_dir, f'*{ext}')))
    image_files = sorted(image_files)

    random.seed(42)
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(0.8 * n_total)
    train_imgs = image_files[:n_train]
    val_imgs = image_files[n_train:]

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
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                yolo_lines = convert_labelme_to_yolo(json_path, (width, height))
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")
                continue

            shutil.copy2(img_path, os.path.join(images_dir, os.path.basename(img_path)))

            label_path = os.path.join(labels_dir, base_name + '.txt')
            if yolo_lines:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
            else:
                open(label_path, 'w').close()

    process_and_save(train_imgs, images_train_dir, labels_train_dir)
    process_and_save(val_imgs, images_val_dir, labels_val_dir)

    # Train a custom CNN model on the processed dataset
    train_custom_cnn(train_imgs, val_imgs, labels_train_dir, labels_val_dir)

    data_yaml = f"""\
path: {root_dir}
train: images/train
val: images/val

nc: 1
names: ['object']
"""
    with open(os.path.join(root_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml)

    train_cmd = [
        'yolo', 'task=detect', 'mode=train',
        'model=yolov8m.pt',
        f'data={os.path.join(root_dir, "data.yaml")}',
        'epochs=3000'
    ]
    print("🚀 Starting YOLOv8 training...")
    try:
        subprocess.run(train_cmd, check=True)
    except Exception as e:
        print("❌ Failed to launch YOLOv8 training:", e)
        return

    # Find the latest training directory and best.pt path
    train_runs = glob('runs/detect/train*')
    if train_runs:
        latest_run = max(train_runs, key=os.path.getmtime)
        best_model_src = os.path.join(latest_run, 'weights', 'best.pt')
    else:
        best_model_src = None

    # Copy best model to root directory and backup in models directory
    if best_model_src and os.path.exists(best_model_src):
        best_model_dst = os.path.join(root_dir, 'best.pt')
        shutil.copy2(best_model_src, best_model_dst)
        models_dir = os.path.join(root_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        backup_model_dst = os.path.join(models_dir, 'best.pt')
        shutil.copy2(best_model_src, backup_model_dst)
    else:
        print("❌ Best model not found after training.")

    # After training, run prediction on validation images and save results
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np

        if not best_model_src or not os.path.exists(best_model_src):
            print(f"❌ Best model not found at {best_model_src}")
            return
        model = YOLO(best_model_src)
        save_dir = 'runs/test_images'
        os.makedirs(save_dir, exist_ok=True)
        print("🚀 Running prediction on validation images...")
        # Run prediction with specified options
        results = model.predict(source=images_val_dir, save=True, save_dir=save_dir,
                                save_txt=True, save_conf=True,
                                vid_stride=1, visualize=False)

        # Replace resized saved images with copies of original quality validation images
        # and overlay predictions manually if necessary
        for result in results:
            # result.orig_img is the original image in numpy array
            # result.path is the path to the input image
            orig_img_path = result.path
            base_name = os.path.basename(orig_img_path)
            saved_img_path = os.path.join(save_dir, base_name)

            # Copy original quality image to replace saved resized image
            shutil.copy2(orig_img_path, saved_img_path)

            # Load original image to overlay predictions
            img = cv2.imread(saved_img_path)
            if img is None:
                continue

            # Overlay boxes and labels from prediction
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{cls} {conf:.2f}"
                # Draw rectangle
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                # Put label
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            # Save the overlaid image
            cv2.imwrite(saved_img_path, img)

        print("✅ Prediction images saved to runs/test_images/")
    except ImportError:
        print("❌ ultralytics package not found. Please install it to run predictions.")
    except Exception as e:
        print(f"❌ Failed to run prediction: {e}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

class CustomCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 80 * 80, 128),  # assuming 640x640 input
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = 1.0 if os.path.getsize(self.label_paths[idx]) > 0 else 0.0
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)

def train_custom_cnn(train_imgs, val_imgs, labels_train_dir, labels_val_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CustomDataset(train_imgs, [os.path.join(labels_train_dir, os.path.splitext(os.path.basename(p))[0] + '.txt') for p in train_imgs], transform)
    val_dataset = CustomDataset(val_imgs, [os.path.join(labels_val_dir, os.path.splitext(os.path.basename(p))[0] + '.txt') for p in val_imgs], transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = CustomCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()