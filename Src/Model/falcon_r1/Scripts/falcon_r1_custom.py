import os
import json
import argparse
import torch
import torch.nn as nn

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Total trainable parameters: {total:,}")

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 🖼️ 4K UHD Resolution
IMG_WIDTH, IMG_HEIGHT = 3840, 2160


class PolygonToBoxDataset(Dataset):
    def __init__(self, img_dir, lbl_dir=None, paired_files=None):
        if paired_files is not None:
            self.img_paths = [pair[0] for pair in paired_files]
            self.lbl_paths = [pair[1] for pair in paired_files]
        else:
            self.img_paths = sorted(
                glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png'))
            )
            if lbl_dir is not None:
                self.lbl_paths = sorted(glob(os.path.join(lbl_dir, '*.json')))
            else:
                self.lbl_paths = []

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.Resize(IMG_HEIGHT, IMG_WIDTH),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if self.lbl_paths:
            lbl_path = self.lbl_paths[idx]
        else:
            lbl_path = None

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img)

        if lbl_path is not None:
            with open(lbl_path, 'r') as f:
                data = json.load(f)

            boxes = []
            for shape in data['shapes']:
                points = np.array(shape['points'])
                x_min = np.min(points[:, 0])
                y_min = np.min(points[:, 1])
                x_max = np.max(points[:, 0])
                y_max = np.max(points[:, 1])
                boxes.append([x_min, y_min, x_max, y_max])
        else:
            boxes = []

        if not boxes:
            box = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        else:
            largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            box = torch.tensor(largest, dtype=torch.float32)

        # Resize image and adjust box
        img_resized = cv2.resize(img_np, (IMG_WIDTH, IMG_HEIGHT))
        scale_x = IMG_WIDTH / w
        scale_y = IMG_HEIGHT / h
        box[0::2] *= scale_x
        box[1::2] *= scale_y

        augmented = self.aug(image=img_resized)
        img_resized = augmented['image']

        return img_resized, box


class StrongDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )

        # Calculate flattened size dynamically for 4K
        feat_w, feat_h = IMG_WIDTH // 32, IMG_HEIGHT // 32
        flattened = feat_w * feat_h * 256

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x)


def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_metrics(model, loader, device, iou_threshold=0.5):
    model.eval()
    TP, FP, FN = 0, 0, 0
    with torch.no_grad():
        for imgs, boxes in loader:
            imgs = imgs.to(device).float() / 255.0
            preds = model(imgs).cpu().numpy()
            gts = boxes.numpy()
            for pred_box, gt_box in zip(preds, gts):
                iou = box_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    TP += 1
                else:
                    FP += 1
                    FN += 1
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"📊 Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")


def train(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, boxes in tqdm(loader, desc="Training"):
        imgs = imgs.to(device).float() / 255.0
        boxes = boxes.to(device)
        preds = model(imgs)
        loss = criterion(preds, boxes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📉 Avg Loss: {total_loss / len(loader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', required=True)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch', '-b', type=int, default=1)  # Start with batch size 1
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', '-d', default='0')
    parser.add_argument('--exp', '-n', default='cnn_4k')
    parser.add_argument('--doot', action='store_true', help='Use doot mode with images and labels in same folder')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model = StrongDetector().to(device)
    count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    def get_loader(split):
        if args.doot:
            all_img_paths = sorted(glob(os.path.join(args.root, '*.jpg')) + glob(os.path.join(args.root, '*.png')))
            paired_files = []
            for img_path in all_img_paths:
                base = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(args.root, base + '.json')
                if os.path.exists(lbl_path):
                    paired_files.append((img_path, lbl_path))
                else:
                    paired_files.append((img_path, None))
            # Split into 90% train, 10% val
            split_idx = int(0.9 * len(paired_files))
            if split == 'train':
                subset_pairs = paired_files[:split_idx]
            elif split == 'val':
                subset_pairs = paired_files[split_idx:]
            else:
                subset_pairs = paired_files
            dataset = PolygonToBoxDataset(None, None, paired_files=subset_pairs)
            return DataLoader(dataset, batch_size=args.batch, shuffle=(split == 'train'))
        else:
            return DataLoader(
                PolygonToBoxDataset(
                    os.path.join(args.root, split, 'images'),
                    os.path.join(args.root, split, 'labels')
                ),
                batch_size=args.batch,
                shuffle=(split == 'train')
            )

    print(f"🚀 Training custom CNN on 4K images in {args.root}")
    if args.doot:
        train_loader = get_loader('train')
        val_loader = get_loader('val')
        test_loader = val_loader
    else:
        train_loader = get_loader('train')
        test_loader = get_loader('test')

    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        train(model, train_loader, device, optimizer, criterion)
        if args.doot:
            evaluate_metrics(model, val_loader, device)
        else:
            evaluate_metrics(model, test_loader, device)

    os.makedirs(f"runs/train/{args.exp}", exist_ok=True)
    torch.save(model.state_dict(), f"runs/train/{args.exp}/best.pt")
    print(f"✅ Model saved to runs/train/{args.exp}/best.pt")

    def save_predictions(model, loader, device, save_dir="runs/test_images", max_samples=10):
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        count = 0
        with torch.no_grad():
            for imgs, boxes in loader:
                imgs = imgs.to(device).float() / 255.0
                preds = model(imgs).cpu().numpy()
                boxes = boxes.cpu().numpy()

                for i in range(len(preds)):
                    pred_box = preds[i]
                    true_box = boxes[i]
                    # Load original image to get DPI
                    img_path = loader.dataset.img_paths[count]
                    orig_img = Image.open(img_path)
                    dpi = orig_img.info.get('dpi', (72, 72))

                    img = orig_img.convert("RGB")
                    draw = ImageDraw.Draw(img)
                    # Draw predicted box in green
                    draw.rectangle([pred_box[0], pred_box[1], pred_box[2], pred_box[3]], outline=(0, 255, 0), width=3)
                    # Draw ground truth box in red
                    draw.rectangle([true_box[0], true_box[1], true_box[2], true_box[3]], outline=(255, 0, 0), width=2)

                    save_path = os.path.join(save_dir, f"pred_{count}.png")
                    img.save(save_path, dpi=dpi)
                    count += 1
                    if count >= max_samples:
                        return

    if args.doot:
        save_predictions(model, val_loader, device)
    else:
        save_predictions(model, test_loader, device)


if __name__ == '__main__':
    main()