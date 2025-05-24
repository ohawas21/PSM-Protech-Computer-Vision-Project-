#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_custom_detector.py

Train a custom CNN object detector on a very small dataset (e.g., 10 images)
using fixed-size input (1200x800) and simplified loss for single-object detection.

Usage:
    python3 train_custom_detector.py --root /path/to/project_root \
        [--epochs 30] [--batch 4] [--lr 1e-4] [--device 0] [--exp custom_cnn]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Fixed image size
IMG_WIDTH, IMG_HEIGHT = 1200, 800
NUM_CLASSES = 1  # single object class

# ───────────────────────────────────────────────────────
# Dataset
class CustomDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.lbl_paths = sorted(glob(os.path.join(lbl_dir, '*.txt')))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        box = []
        with open(self.lbl_paths[idx], 'r') as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw / 2) * IMG_WIDTH)
                y1 = int((y - bh / 2) * IMG_HEIGHT)
                x2 = int((x + bw / 2) * IMG_WIDTH)
                y2 = int((y + bh / 2) * IMG_HEIGHT)
                box = [x1, y1, x2, y2]
                break  # only first box assumed
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(box, dtype=torch.float32)


# ───────────────────────────────────────────────────────
# Model
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
            nn.Linear(256, 4)  # bounding box x1, y1, x2, y2
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ───────────────────────────────────────────────────────
# Training and Evaluation
def train(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, boxes in tqdm(loader, desc="Training"):
        imgs, boxes = imgs.to(device).float() / 255.0, boxes.to(device)
        preds = model(imgs)
        loss = criterion(preds, boxes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📉 Average Loss: {total_loss / len(loader):.4f}")


def evaluate(model, loader, device):
    model.eval()
    print("🔍 Sample Predictions:")
    with torch.no_grad():
        for imgs, boxes in loader:
            imgs = imgs.to(device).float() / 255.0
            preds = model(imgs).cpu().numpy()
            print("Pred:", preds[0])
            print("True:", boxes[0].numpy())
            break


# ───────────────────────────────────────────────────────
# CLI Wrapper
def main():
    parser = argparse.ArgumentParser(description="Train a custom CNN object detector")
    parser.add_argument('--root', '-r', required=True, help='Project root directory')
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
            CustomDataset(
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

    # Save model
    os.makedirs(f"runs/train/{args.exp}", exist_ok=True)
    torch.save(model.state_dict(), f"runs/train/{args.exp}/best.pt")
    print(f"✅ Model saved at runs/train/{args.exp}/best.pt")


if __name__ == '__main__':
    main()