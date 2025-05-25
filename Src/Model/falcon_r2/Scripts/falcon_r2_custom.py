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
    Looks for `img.stem + .json` first, then `img.name + .json`.
    Raises FileNotFoundError if neither exists.
    """
    cand1 = img_path.with_suffix(".json")
    cand2 = img_path.with_suffix(img_path.suffix + ".json")
    json_path = cand1 if cand1.exists() else cand2
    if not json_path.exists():
        raise FileNotFoundError

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    boxes, labels = [], []
    for shp in data.get("shapes", []):
        xs, ys = zip(*shp["points"])
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        labels.append(shp["label"])
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

def build_filelists(data_dir: Path, exts: Tuple[str, ...]):
    imgs = [p for e in exts for p in data_dir.glob(f"*.{e.lstrip('.')}")]
    if not imgs:
        raise FileNotFoundError(f"No images with extensions {exts} in {data_dir}")

    valid, first_labels = [], []
    for p in imgs:
        try:
            boxes, labels = _extract_boxes(p)
        except FileNotFoundError:
            log.warning("skip %s : no JSON", p.name)
            continue
        valid.append(p)
        first_labels.append(labels[0] if labels else "_background_")

    if not valid:
        raise RuntimeError("After skipping missing annotations, dataset is empty")
    return valid, first_labels


def stratified(ids: List[int], labels: List[str], ratio=0.2):
    if len(set(labels)) < 2:
        return train_test_split(ids, test_size=ratio, random_state=42)
    sss = StratifiedShuffleSplit(1, test_size=ratio, random_state=42)
    return next(sss.split(ids, labels))


def build_datasets(root: Path, img_size: int, exts: Tuple[str, ...]):
    files, lbls = build_filelists(root, exts)
    classes = sorted({l for l in lbls if l != "_background_"})
    label2idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background

    ids = list(range(len(files)))
    tr_ids, va_ids = stratified(ids, lbls)
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    tr_ds = LabelMeDetDataset([files[i] for i in tr_ids], label2idx, tfm)
    va_ds = LabelMeDetDataset([files[i] for i in va_ids], label2idx, tfm)
    return tr_ds, va_ds, label2idx

# ─────────────────────────────────── main ──────────────────────────────────────

def main(cfg):
    data_dir = Path(cfg.data_dir)
    exts = tuple(e.strip().lstrip(".") for e in cfg.img_exts.split(","))

    tr_ds, va_ds, l2i = build_datasets(data_dir, cfg.image_size, exts)
    num_classes = len(l2i) + 1

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = Detector(num_classes, lr=cfg.lr, crop_dir=Path(cfg.output_dir))
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val/mAP50", mode="max", save_top_k=1,
                                        filename="det-{epoch:02d}-{val/mAP50:.2f}")
    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator="auto", devices="auto",
                         precision=16 if torch.cuda.is_available() else 32,
                         callbacks=[ckpt], log_every_n_steps=10)

    trainer.fit(model, tr_loader, va_loader)
    trainer.test(model, va_loader, ckpt_path="best")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Faster-R-CNN + auto-crop for LabelMe datasets")
    ap.add_argument("--data_dir", required=True, help="Folder with images + LabelMe JSONs")
    ap.add_argument("--output_dir", default="./crops")
    ap.add_argument("--img_exts", default="jpg,jpeg,png", help="Comma-separated list of image extensions")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=640)
    main(ap.parse_args())
