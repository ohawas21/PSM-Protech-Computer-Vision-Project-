# falcon_r2_custom.py
"""
Faster R-CNN (ResNet-50 FPN) + Auto-Crop for LabelMe Datasets
============================================================
* **Zero-friction ingest**: `--img_exts jpg,png` (default) – any mixture allowed.
* **LabelMe polygons ➜ axis-aligned bboxes** handled transparently.
* **Robust train/val split**: stratified when ≥ 2 classes, plain otherwise.
* **Missing annotation skip**: images without JSON are logged & ignored – training never aborts.
* **Post-fit crop export**: every detection ≥ 0.50 confidence saved to `--output_dir`.

Install once
------------
```bash
pip install torch torchvision torchmetrics pytorch-lightning==2.2.0 scikit-learn
```
Run
---
```bash
python falcon_r2_custom.py \
       --data_dir ../Dataset \
       --output_dir ./crops \
       --img_exts png          # optional
```
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("falcon_r2")

# ─────────────────────────────── util: LabelMe → bbox ──────────────────────────

def _extract_boxes(img_path: Path) -> Tuple[List[List[float]], List[str]]:
    """Return axis-aligned boxes & labels for *one* image.
    Looks for multiple JSON naming patterns:
    1. img.stem + .json (e.g., 80.png -> 80.json)
    2. img.name + .json (e.g., 80.png -> 80.png.json)
    3. Same directory structure but in different folders
    
    Raises FileNotFoundError if no JSON is found.
    """
    # Pattern 1: Replace extension with .json (most common)
    cand1 = img_path.with_suffix(".json")
    
    # Pattern 2: Add .json to full filename
    cand2 = img_path.with_suffix(img_path.suffix + ".json")
    
    # Check if either exists
    json_path = None
    if cand1.exists():
        json_path = cand1
    elif cand2.exists():
        json_path = cand2
    else:
        # Log the specific files we looked for
        log.debug(f"Checked for annotations: {cand1}, {cand2}")
        raise FileNotFoundError(f"No JSON annotation found for {img_path.name}")

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        log.warning(f"Invalid JSON in {json_path}: {e}")
        raise FileNotFoundError(f"Invalid JSON format in {json_path}")

    boxes, labels = [], []
    shapes = data.get("shapes", [])
    
    if not shapes:
        log.warning(f"No shapes found in {json_path}")
        return boxes, labels
    
    for shp in shapes:
        points = shp.get("points", [])
        label = shp.get("label", "unknown")
        
        if len(points) < 2:
            log.warning(f"Skipping shape with insufficient points in {json_path}")
            continue
            
        try:
            xs, ys = zip(*points)
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(label)
        except (ValueError, TypeError) as e:
            log.warning(f"Skipping malformed shape in {json_path}: {e}")
            continue
    
    return boxes, labels

# ─────────────────────────────────── Dataset ───────────────────────────────────
class LabelMeDetDataset(Dataset):
    def __init__(self, files: Sequence[Path], label2idx: Dict[str, int], tfm: T.Compose):
        self.files = list(files)
        self.label2idx = label2idx
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        boxes, labels_str = _extract_boxes(img_path)
        labels = [self.label2idx[l] for l in labels_str]

        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)
        tgt = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, tgt, img_path.name


def collate_fn(batch):
    imgs, tgts, names = zip(*batch)
    return list(imgs), list(tgts), list(names)

# ────────────────────────────── Lightning module ───────────────────────────────
class Detector(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float, crop_dir: Path):
        super().__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=num_classes)
        self.lr = lr
        self.crop_dir = crop_dir
        self.map = MeanAveragePrecision(iou_type="bbox")
        self.macro_prec = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    # — training —
    def training_step(self, batch, _):
        imgs, tgts, _ = batch
        loss_dict = self.model(imgs, tgts)
        loss = sum(loss_dict.values())
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        return loss

    # — validation —
    def validation_step(self, batch, _):
        imgs, tgts, _ = batch
        preds = self.model(imgs)
        self.map.update(preds, tgts)
        self.macro_prec.update(preds, tgts)

    def on_validation_epoch_end(self):
        m = self.map.compute(); p = self.macro_prec.compute()
        self.log("val/mAP50", m["map_50"], prog_bar=True)
        self.log("val/macro_prec", p["precision"].mean(), prog_bar=True)
        self.map.reset(); self.macro_prec.reset()

    # — test / crop —
    def test_step(self, batch, _):
        imgs, _, names = batch
        preds = self.model(imgs)
        self._export_crops(imgs, preds, names, thr=0.5)

    @torch.inference_mode()
    def _export_crops(self, imgs, preds, names, thr):
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for im, pr, n in zip(imgs, preds, names):
            pil = to_pil(im.cpu())
            for i, s in enumerate(pr["scores"]):
                if s < thr: break
                x1, y1, x2, y2 = map(int, pr["boxes"][i])
                pil.crop((x1, y1, x2, y2)).save(self.crop_dir / f"{Path(n).stem}_crop_{i}.jpg")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[16, 22], gamma=0.1)
        return [opt], [sch]

# ───────────────────────────── dataset helpers ─────────────────────────────────