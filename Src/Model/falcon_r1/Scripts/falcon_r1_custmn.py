import os
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

# Dataset for polygon-based YOLO labels
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
            points = np.array(coords[1:]).reshape(-1, 2)
            x_min = np.min(points[:, 0]) * IMG_WIDTH
            y_min = np.min(points[:, 1]) * IMG_HEIGHT
            x_max = np.max(points[:, 0]) * IMG_WIDTH
            y_max = np.max(points[:, 1]) * IMG_HEIGHT
            box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        return img, box


# Stronger CNN architecture
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
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear((IMG_WIDTH // 32) * (IMG_HEIGHT // 32) * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x)


# Metrics calculation
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


# Training loop
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


# CLI Entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', required=True)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', '-d', default='0')
    parser.add_argument('--exp', '-n', default='custom_cnn')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model = StrongDetector().to(device)
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
        evaluate_metrics(model, test_loader, device)

    os.makedirs(f"runs/train/{args.exp}", exist_ok=True)
    torch.save(model.state_dict(), f"runs/train/{args.exp}/best.pt")
    print(f"✅ Model saved to runs/train/{args.exp}/best.pt")


if __name__ == '__main__':
    main()