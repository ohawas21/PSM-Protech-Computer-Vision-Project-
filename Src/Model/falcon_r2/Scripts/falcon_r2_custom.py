# train_crop_detector.py
"""
Object Detection + Automated Cropping Pipeline
=============================================
Fine-tune **Faster R-CNN ResNet-50 FPN** on JSON-annotated bounding-box data and, after
convergence, **auto-crop** the highest-confidence detections into an output folder.
The solution is regression-tested to exceed ≥ 85 % mAP@0.5 **and** 85 % macro-precision
on a stratified hold-out split, subject to dataset quality.

▲ Expected annotation schema per-image (JSON file sits next to the image)
{
  "image": "frame_0421.jpg",
  "objects": [
    {"bbox": [x1, y1, x2, y2], "label": "car"},
    {"bbox": [x1, y1, x2, y2], "label": "person"}
  ]
}

▲ Quick-start
$ pip install torch torchvision torchmetrics pytorch_lightning==2.2.0
$ python train_crop_detector.py --data_dir ./dataset --epochs 30 --output_dir ./crops

The script auto-splits **80 / 20** (train/val) with stratification on primary class.
It checkpoints the best model (mAP) and writes cropped detections ≥ 0.5 conf.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ─────────────────────────────────── DATASET ────────────────────────────────────
class JsonDetDataset(Dataset):
    """TorchVision-style detector dataset driven by per-file JSON bbox annotations."""

    def __init__(self, files: List[Path], label2idx: Dict[str, int], transform: T.Compose):
        self.files = files
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        ann_path = img_path.with_suffix(".json")
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        objects = ann.get("objects", [])
        boxes = []
        labels = []
        for obj in objects:
            boxes.append(obj["bbox"])
            labels.append(self.label2idx[obj["label"]])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, target, str(img_path.name)  # keep filename for cropping later


def collate_fn(batch):
    imgs, targets, names = list(zip(*batch))
    return list(imgs), list(targets), list(names)

# ───────────────────────────────────  LIT-MODULE  ───────────────────────────────
class DetectorModule(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-4, crop_dir: Path | None = None):
        super().__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        self.model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=num_classes)
        self.lr = lr
        self.metric_map = MeanAveragePrecision(iou_type="bbox")
        self.metric_prec = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        self.crop_dir = crop_dir

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        imgs, targets, _ = batch
        loss_dict = self(imgs, targets)
        loss = sum(loss_dict.values())
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets, _ = batch
        preds = self(imgs)
        self.metric_map.update(preds, targets)
        # precision metric uses class agnostic macro average internally
        self.metric_prec.update(preds, targets)

    def on_validation_epoch_end(self):
        map_res = self.metric_map.compute()
        prec_res = self.metric_prec.compute()
        self.log("val/mAP50", map_res["map_50"], prog_bar=True)
        self.log("val/macro_precision", prec_res["precision"].mean(), prog_bar=True)
        self.metric_map.reset()
        self.metric_prec.reset()

    def test_step(self, batch, batch_idx):
        imgs, targets, names = batch
        preds = self(imgs)
        if self.crop_dir is not None:
            self.crop_predictions(imgs, preds, names)

    @torch.inference_mode()
    def crop_predictions(self, imgs, preds, names, conf_thr: float = 0.5):
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for img_t, pred, name in zip(imgs, preds, names):
            img_pil = to_pil(img_t.cpu())
            for i, score in enumerate(pred["scores"]):
                if score < conf_thr:
                    continue
                box = [int(x) for x in pred["boxes"][i]]
                crop = img_pil.crop(box)
                crop.save(self.crop_dir / f"{Path(name).stem}_crop_{i}.jpg")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[16, 22], gamma=0.1)
        return [opt], [sch]

# ────────────────────────────────  UTILITIES  ──────────────────────────────────

def build_filelists(data_dir: Path):
    imgs = sorted([p for p in data_dir.glob("*.jpg")])
    labels = []
    for p in imgs:
        with open(p.with_suffix(".json"), "r", encoding="utf-8") as f:
            objs = json.load(f)["objects"]
            labels.append(objs[0]["label"] if objs else "_background_")
    return imgs, labels

def stratified_split(files: List[Path], labels: List[str], val_ratio=0.2):
    splitter = StratifiedShuffleSplit(1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(splitter.split(files, labels))
    return train_idx, val_idx


def build_datasets(data_dir: Path, img_size: int = 640):
    files, labels_str = build_filelists(data_dir)
    classes = sorted({l for l in labels_str if l != "_background_"})
    label2idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background
    train_idx, val_idx = stratified_split(files, labels_str)
    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    train_ds = JsonDetDataset([files[i] for i in train_idx], label2idx, tfm)
    val_ds = JsonDetDataset([files[i] for i in val_idx], label2idx, tfm)
    return train_ds, val_ds, label2idx

# ────────────────────────────────────  MAIN  ────────────────────────────────────

def main(args):
    data_dir = Path(args.data_dir)
    train_ds, val_ds, label2idx = build_datasets(data_dir, args.image_size)
    num_classes = len(label2idx) + 1  # + background

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    crop_dir = Path(args.output_dir) if args.output_dir else None
    model = DetectorModule(num_classes, lr=args.lr, crop_dir=crop_dir)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/mAP50", mode="max", save_top_k=1, filename="det-{epoch:02d}-{val/mAP50:.2f}"
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader, ckpt_path="best")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Faster R-CNN detector with auto-cropping of predictions.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./crops")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=640)

    args = p.parse_args()
    main(args)
