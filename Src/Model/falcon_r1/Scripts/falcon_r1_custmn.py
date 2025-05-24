#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_custom_detector.py

Train a custom CNN object detector on polygon-annotated data by converting
polygons to bounding boxes. For use with very small datasets (e.g., 10 images).

Usage:
    python3 train_custom_detector.py --root /path/to/project_root \
        [--epochs 30] [--batch 4] [--lr 1e-4] [--device 0] [--exp custom_cnn]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMG_WIDTH, IMG_HEIGHT = 1200, 800
NUM_CLASSES = 1  # only one class

# ───────────────────────────────────────────────────────
# Dataset class supporting polygon to box
class PolygonToBoxDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.lbl_paths = sorted(glob(os.path.join(lbl_dir, '*.txt')))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        with open(self.lbl_paths[idx], 'r') as f:
            line = f.readline().strip()
            coords = list(map(float, line.strip().split()))
            cls_id = int(coords[0])
            points = np.array(coords[1:]).reshape(-1, 2)
            x_min = np.min(points[:, 0]) * IMG_WIDTH
            y_min = np.min(points[:, 1]) * IMG_HEIGHT
            x_max = np.max(points[:, 0]) * IMG_WIDTH
            y_max = np.max(points[:, 1]) * IMG_HEIGHT
            box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        return img, box


# ───────────────────────────────────────────────────────
# Simple CNN Detector
class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear((IMG_WIDTH // 8) * (IMG_HEIGHT // 8) * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # x1, y1, x2, y2
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ───────────────────────────────────────────────────────
# Train and Evaluate
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


def evaluate(model, loader, device):
    model.eval()
    print("🔍 Sample Predictions:")
    with torch.no_grad():
        for imgs, boxes in loader:
            imgs = imgs.to(device).float() / 255.0
            preds = model(imgs).cpu().numpy()
            print("Pred box:", preds[0])
            print("True box:", boxes[0].numpy())
            break


# ───────────────────────────────────────────────────────
# Main CLI
def main():
    parser = argparse.ArgumentParser(description="Train a CNN detector on polygon labels")
    parser.add_argument('--root', '-r', required=True, help='Dataset root directory')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', '-d', type=str, default='0')
    parser.add_argument('--exp', '-n', default='custom_cnn', help='Experiment name')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model = SimpleDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    def get_loader(split):
        return DataLoader(
            PolygonToBoxDataset(
                os.path.join(args.root, split, 'images'),
                os.path.join(args.root, split, 'labels'),
                transform=T.ToTensor()
            ),
            batch_size=args.batch,
            shuffle=(split == 'train')
        )

    print(f"🚀 Training custom CNN on {args.root}")
    train_loader = get_loader('train')
    test_loader = get_loader('test')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train(model, train_loader, device, optimizer, criterion)
        evaluate(model, test_loader, device)

    os.makedirs(f"runs/train/{args.exp}", exist_ok=True)
    torch.save(model.state_dict(), f"runs/train/{args.exp}/best.pt")
    print(f"✅ Model saved to runs/train/{args.exp}/best.pt")


if __name__ == '__main__':
    main()