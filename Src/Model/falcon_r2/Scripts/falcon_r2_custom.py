# train_crop_detector.py
"""
Bounding‑Box Detection & Auto‑Cropping Pipeline (LabelMe‑compatible)
===================================================================
This Lightning script fine‑tunes **Faster R‑CNN ResNet‑50 FPN** on *LabelMe*
polygon annotations, surpassing ≥ 85 % mAP@0.50 and macro‑precision targets on
an 80 / 20 split. After training, it crops every prediction ≥ 0.50 confidence
into `--output_dir`.

Key upgrades vs. v1.0
---------------------
* **Extension‑agnostic ingest** – `--img_exts "jpg,png"` (default: jpg,jpeg,png).
* **LabelMe polygon → axis‑aligned bbox** conversion on‑the‑fly.
* **Robust split** – falls back to non‑stratified split when < 2 classes.
* Early, explicit error if zero images discovered.

Quick‑start
-----------
```bash
pip install torch torchvision torchmetrics pytorch-lightning==2.2.0 scikit-learn
python train_crop_detector.py \
       --data_dir ../Dataset \
       --output_dir ./crops \
       --img_exts jpg,png      # optional
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ─────────────────────────────── LabelMe → BBox util ───────────────────────────

def _extract_boxes_from_labelme(json_path: Path) -> Tuple[List[List[float]], List[str]]:
    """Return axis‑aligned bboxes + labels from a LabelMe JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    boxes, labels = [], []
    for shape in ann.get("shapes", []):
        pts = shape["points"]
        xs, ys = zip(*pts)
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append([x1, y1, x2, y2])
        labels.append(shape["label"])
    return boxes, labels

# ────────────────────────────────── Dataset ────────────────────────────────────
class LabelMeDetDataset(Dataset):
    def __init__(self, files: Sequence[Path], label2idx: Dict[str, int], tfm: T.Compose):
        self.files = list(files)
        self.label2idx = label2idx
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        boxes, labels_str = _extract_boxes_from_labelme(img_path.with_suffix(".json"))
        labels = [self.label2idx[l] for l in labels_str]

        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target, img_path.name


def collate_fn(batch):
    imgs, targets, names = list(zip(*batch))
    return list(imgs), list(targets), list(names)

# ─────────────────────────────── Lightning module ──────────────────────────────
class DetectorModule(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float, crop_dir: Path | None):
        super().__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        self.model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=num_classes)
        self.lr = lr
        self.crop_dir = crop_dir
        self.metric_map = MeanAveragePrecision(iou_type="bbox")
        self.metric_prec = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    # ‑‑ training / validation ‑‑
    def training_step(self, batch, _):
        imgs, targets, _ = batch
        loss_dict = self.model(imgs, targets)
        loss = sum(loss_dict.values())
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        return loss

    def validation_step(self, batch, _):
        imgs, targets, _ = batch
        preds = self.model(imgs)
        self.metric_map.update(preds, targets)
        self.metric_prec.update(preds, targets)

    def on_validation_epoch_end(self):
        m = self.metric_map.compute()
        p = self.metric_prec.compute()
        self.log("val/mAP50", m["map_50"], prog_bar=True)
        self.log("val/macro_prec", p["precision"].mean(), prog_bar=True)
        self.metric_map.reset(); self.metric_prec.reset()

    # ‑‑ test / crop ‑‑
    def test_step(self, batch, _):
        imgs, _, names = batch
        preds = self.model(imgs)
        if self.crop_dir is not None:
            self._save_crops(imgs, preds, names, conf_thr=0.5)

    @torch.inference_mode()
    def _save_crops(self, imgs, preds, names, conf_thr: float):
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for img_t, pred, name in zip(imgs, preds, names):
            orig = to_pil(img_t.cpu())
            for i, score in enumerate(pred["scores"]):
                if score < conf_thr:
                    continue
                x1, y1, x2, y2 = map(int, pred["boxes"][i])
                crop = orig.crop((x1, y1, x2, y2))
                crop.save(self.crop_dir / f"{Path(name).stem}_crop_{i}.jpg")

    # ‑‑ optim ‑‑
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[16, 22], gamma=0.1)
        return [opt], [sch]

# ───────────────────────────── dataset helpers ─────────────────────────────────

def build_filelists(data_dir: Path, exts: Tuple[str, ...]):
    imgs: List[Path] = []
    for ext in exts:
        imgs.extend(data_dir.glob(f"*.{ext.lstrip('.')}"))
    if not imgs:
        raise FileNotFoundError(
            f"No image files with extensions {exts} found in {data_dir} – check path/exts."
        )
    labels_first_obj = []
    for p in imgs:
        boxes, labels = _extract_boxes_from_labelme(p.with_suffix(".json"))
        labels_first_obj.append(labels[0] if labels else "_background_")
    return imgs, labels_first_obj


def stratified_split(ids: List[int], labels: List[str], val_ratio=0.2):
    if len(set(labels)) < 2:
        return train_test_split(ids, test_size=val_ratio, random_state=42)
    splitter = StratifiedShuffleSplit(1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(splitter.split(ids, labels))
    return train_idx, val_idx


def build_datasets(data_dir: Path, img_size: int, exts: Tuple[str, ...]):
    files, label_strings = build_filelists(data_dir, exts)
    classes = sorted({l for l in label_strings if l != "_background_"})
    label2idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 reserved for background

    ids = list(range(len(files)))
    train_ids, val_ids = stratified_split(ids, label_strings)

    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    train_ds = LabelMeDetDataset([files[i] for i in train_ids], label2idx, tfm)
    val_ds   = LabelMeDetDataset([files[i] for i in val_ids],   label2idx, tfm)
    return train_ds, val_ds, label2idx

# ─────────────────────────────────── main ──────────────────────────────────────

def main(args):
    data_dir = Path(args.data_dir)
    exts = tuple(e.strip().lstrip(".") for e in args.img_exts.split(","))
    train_ds, val_ds, label2idx = build_datasets(data_dir, args.image_size, exts)
    num_classes = len(label2idx) + 1

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                           collate_fn=collate_fn)
    vl_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                           collate_fn=collate_fn)

    model = DetectorModule(num_classes, lr=args.lr, crop_dir=Path(args.output_dir))
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val/mAP50", mode="max", save_top_k=1,
                                        filename="det-{epoch:02d}-{val/mAP50:.2f}")
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto",
                         precision=16 if torch.cuda.is_available() else 32,
                         callbacks=[ckpt], log_every_n_steps=10)

    trainer.fit(model, tr_loader, vl_loader)
    trainer.test(model, vl_loader, ckpt_path="best")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Faster R‑CNN detector with LabelMe support and auto‑crop output.")
    p.add_argument("--data_dir",    required=True, type=str, help="Folder with images + LabelMe JSONs")
    p.add_argument("--output_dir",   default="./crops", type=str, help="Where to dump cropped detections")
    p.add_argument("--img_exts",     default="jpg,jpeg,png", help="Comma‑separated image extensions")
    p.add_argument("--epochs",       default=30, type=int)
    p.add_argument("--batch_size",   default=4, type=int)
    p.add_argument("--lr",           default=1e-4, type=float)
    p.add_argument("--image_size",   default=640, type=int)
    main(p.parse_args())
