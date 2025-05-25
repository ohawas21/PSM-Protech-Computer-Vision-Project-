import os, json, argparse, torch, torch.nn as nn
import numpy as np, cv2
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet50
from torch.cuda.amp import autocast, GradScaler

IMG_WIDTH, IMG_HEIGHT = 1920, 1080  # ↓ Reduced from 4K

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Total trainable parameters: {total:,}")

class PolygonToBoxDataset(Dataset):
    def __init__(self, img_dir=None, lbl_dir=None, paired_files=None):
        if paired_files:
            self.img_paths = [p[0] for p in paired_files]
            self.lbl_paths = [p[1] for p in paired_files]
        else:
            self.img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png')))
            self.lbl_paths = sorted(glob(os.path.join(lbl_dir, '*.json'))) if lbl_dir else []

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3), A.Rotate(limit=30, p=0.5),
            A.Resize(IMG_HEIGHT, IMG_WIDTH), ToTensorV2()
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, lbl_path = self.img_paths[idx], self.lbl_paths[idx] if self.lbl_paths else None
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img)

        boxes = []
        if lbl_path:
            try:
                with open(lbl_path, 'r') as f:
                    data = json.load(f)
                for shape in data['shapes']:
                    pts = np.array(shape['points'])
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)
                    boxes.append([x_min, y_min, x_max, y_max])
            except Exception as e:
                print(f"⚠️ Warning: Failed to load or parse JSON {lbl_path}: {e}")
                boxes = []

        box = torch.tensor([0, 0, 1, 1], dtype=torch.float32) if not boxes else \
              torch.tensor(max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])), dtype=torch.float32)

        img_resized = cv2.resize(img_np, (IMG_WIDTH, IMG_HEIGHT))
        box[0::2] *= IMG_WIDTH / w
        box[1::2] *= IMG_HEIGHT / h

        return self.aug(image=img_resized)['image'], box

class StrongDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights='DEFAULT')
        for param in backbone.parameters(): param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 4)
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return self.regressor(x)

def box_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def giou_loss(preds, targets):
    # preds and targets: [batch_size, 4] (x_min, y_min, x_max, y_max)
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
    x_min_tgt, y_min_tgt, x_max_tgt, y_max_tgt = targets[:,0], targets[:,1], targets[:,2], targets[:,3]

    # Intersection
    x_min_inter = torch.max(x_min_pred, x_min_tgt)
    y_min_inter = torch.max(y_min_pred, y_min_tgt)
    x_max_inter = torch.min(x_max_pred, x_max_tgt)
    y_max_inter = torch.min(y_max_pred, y_max_tgt)

    inter_w = (x_max_inter - x_min_inter).clamp(min=0)
    inter_h = (y_max_inter - y_min_inter).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    area_tgt = (x_max_tgt - x_min_tgt) * (y_max_tgt - y_min_tgt)
    union_area = area_pred + area_tgt - inter_area

    iou = inter_area / (union_area + 1e-7)

    # Enclosing box
    x_min_enc = torch.min(x_min_pred, x_min_tgt)
    y_min_enc = torch.min(y_min_pred, y_min_tgt)
    x_max_enc = torch.max(x_max_pred, x_max_tgt)
    y_max_enc = torch.max(y_max_pred, y_max_tgt)

    enc_w = (x_max_enc - x_min_enc).clamp(min=0)
    enc_h = (y_max_enc - y_min_enc).clamp(min=0)
    enc_area = enc_w * enc_h + 1e-7

    giou = iou - (enc_area - union_area) / enc_area
    loss = 1 - giou
    return loss.mean()

def evaluate_metrics(model, loader, device, iou_threshold=0.5):
    model.eval()
    TP = FP = FN = 0
    with torch.no_grad():
        for imgs, boxes in loader:
            imgs = imgs.to(device).float() / 255.0
            with autocast():
                preds = model(imgs).cpu().numpy()
            gts = boxes.numpy()
            for p, g in zip(preds, gts):
                iou = box_iou(p, g)
                if iou >= iou_threshold: TP += 1
                else: FP += 1; FN += 1
    p = TP / (TP + FP + 1e-6); r = TP / (TP + FN + 1e-6); f1 = 2*p*r / (p+r+1e-6)
    print(f"📊 Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

def train(model, loader, device, optimizer, criterion, scaler):
    model.train(); total_loss = 0
    for imgs, boxes in tqdm(loader, desc="Training"):
        imgs, boxes = imgs.to(device).float() / 255.0, boxes.to(device)
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss = criterion(preds, boxes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    print(f"📉 Avg Loss: {total_loss / len(loader):.4f}")

def save_predictions(model, loader, device, save_dir="runs/test_images"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (imgs, boxes) in enumerate(loader):
            imgs = imgs.to(device).float() / 255.0
            with autocast():
                preds = model(imgs).cpu().numpy()
            boxes = boxes.cpu().numpy()
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                img_path = loader.dataset.img_paths[idx * loader.batch_size + i]
                orig_img = Image.open(img_path).convert("RGB")
                dpi = orig_img.info.get('dpi', (72, 72))
                draw = ImageDraw.Draw(orig_img)
                draw.rectangle(preds[i].tolist(), outline="green", width=4)
                draw.rectangle(boxes[i].tolist(), outline="red", width=3)
                base_name = os.path.basename(img_path)
                save_path = os.path.join(save_dir, base_name)
                orig_img.save(save_path, dpi=dpi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--exp', default='cnn_halfres')
    parser.add_argument('--doot', action='store_true')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model = StrongDetector().to(device); count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = giou_loss
    scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def unfreeze_last_resnet_block(model):
        # Unfreeze layer4 of resnet50 backbone
        for name, param in model.feature_extractor.named_parameters():
            if name.startswith('6.'):  # layer4 is the 7th child (index 6)
                param.requires_grad = True

    def get_loader(split):
        if args.doot:
            img_paths = sorted(glob(os.path.join(args.root, '*.jpg')) + glob(os.path.join(args.root, '*.png')))
            pairs = []
            for p in img_paths:
                label_path = os.path.splitext(p)[0] + '.json'
                if os.path.exists(label_path):
                    pairs.append((p, label_path))
            split_idx = int(0.9 * len(pairs))
            subset = pairs[:split_idx] if split == 'train' else pairs[split_idx:]
            return DataLoader(PolygonToBoxDataset(paired_files=subset), batch_size=args.batch, shuffle=(split == 'train'))
        else:
            return DataLoader(
                PolygonToBoxDataset(
                    os.path.join(args.root, split, 'images'),
                    os.path.join(args.root, split, 'labels')),
                batch_size=args.batch, shuffle=(split == 'train'))

    train_loader = get_loader('train')
    val_loader = get_loader('val') if args.doot else get_loader('test')

    best_loss = float('inf')
    patience = 10
    trigger_times = 0

    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")

        # Unfreeze last ResNet block after 10 epochs
        if epoch == 10:
            unfreeze_last_resnet_block(model)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        train(model, train_loader, device, optimizer, criterion, scaler)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, boxes in val_loader:
                imgs, boxes = imgs.to(device).float() / 255.0, boxes.to(device)
                preds = model(imgs)
                loss = criterion(preds, boxes)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"🔍 Validation Loss: {val_loss:.4f}")

        evaluate_metrics(model, val_loader, device)

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
            os.makedirs(f"runs/train/{args.exp}", exist_ok=True)
            torch.save(model.state_dict(), f"runs/train/{args.exp}/best.pt")
            print(f"✅ Saved to runs/train/{args.exp}/best.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"⏹ Early stopping triggered after {patience} epochs with no improvement.")
                break

    save_predictions(model, val_loader, device)

if __name__ == "__main__":
    main()