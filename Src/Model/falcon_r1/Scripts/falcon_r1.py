#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train YOLOv11-L object detector with grayscale augmentation (15%)
on 1200×800 images, using your existing folder structure:
  project_root/
    data.yaml
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/
"""

import os
import sys
from ultralytics import YOLO

def main():
    # ─── CONFIG ──────────────────────────────────────────────────────────────────
    # Change these paths if your structure lives elsewhere
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA_YAML    = os.path.join(PROJECT_ROOT, 'data.yaml')

    # Folders for splits
    TRAIN_IMAGES = os.path.join(PROJECT_ROOT, 'train', 'images')
    TRAIN_LABELS = os.path.join(PROJECT_ROOT, 'train', 'labels')
    VAL_IMAGES   = os.path.join(PROJECT_ROOT, 'val',   'images')
    VAL_LABELS   = os.path.join(PROJECT_ROOT, 'val',   'labels')
    TEST_IMAGES  = os.path.join(PROJECT_ROOT, 'test',  'images')
    TEST_LABELS  = os.path.join(PROJECT_ROOT, 'test',  'labels')

    # YOLO & training Hyperparameters
    MODEL_NAME   = 'yolov11l.pt'     # or path to your custom YOLOv11L weights
    EPOCHS       = 50
    BATCH_SIZE   = 8                 # reduce if you hit OOM
    IMGSZ        = (1200, 800)       # (width, height)
    DEVICE       = '0'               # GPU index (or 'cpu')
    EXP_NAME     = 'yolov11l_gray15' # run folder name

    # Augmentation parameters
    AUG_KWARGS = dict(
        mosaic      = True,
        mixup       = 0.15,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 2.0,
        translate   = 0.08,
        scale       = 0.5,
        shear       = 0.0,
        perspective = 0.0,
        flipud      = 0.0,
        fliplr      = 0.5,
        grayscale   = 0.15,  # 15% probability
    )
    # ──────────────────────────────────────────────────────────────────────────────


    # ─── SANITY CHECKS ─────────────────────────────────────────────────────────────
    # Ensure data.yaml is present
    if not os.path.isfile(DATA_YAML):
        sys.exit(f"❌ data.yaml not found at {DATA_YAML}")

    # Ensure each split has both images and labels subfolders
    for split in ('train', 'val', 'test'):
        img_dir = os.path.join(PROJECT_ROOT, split, 'images')
        lbl_dir = os.path.join(PROJECT_ROOT, split, 'labels')
        if not os.path.isdir(img_dir):
            sys.exit(f"❌ Missing image folder: {img_dir}")
        if not os.path.isdir(lbl_dir):
            sys.exit(f"❌ Missing label folder: {lbl_dir}")
    # ──────────────────────────────────────────────────────────────────────────────


    # ─── TRAINING ─────────────────────────────────────────────────────────────────
    print("🚀 Starting YOLOv11-L training with grayscale augmentation (15%)")
    print(f"• Project root : {PROJECT_ROOT}")
    print(f"• data.yaml    : {DATA_YAML}")
    print(f"• Model        : {MODEL_NAME}")
    print(f"• Epochs       : {EPOCHS}")
    print(f"• Batch size   : {BATCH_SIZE}")
    print(f"• Image size   : {IMGSZ}")
    print(f"• Device       : {DEVICE}")
    print(f"• Experiment   : runs/train/{EXP_NAME}")

    # Load YOLOv11 model
    model = YOLO(MODEL_NAME)

    # Train with augmentations
    results = model.train(
        data         = DATA_YAML,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH_SIZE,
        device       = DEVICE,
        project      = PROJECT_ROOT,
        name         = EXP_NAME,
        exist_ok     = False,     # error if 'runs/train/EXP_NAME' exists
        save         = True,
        save_period  = -1,        # only best + final
        augment      = True,
        augment_kwargs = AUG_KWARGS,
    )

    # Report best checkpoint
    best_ckpt = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"\n✅ Training complete! Best model: {best_ckpt}")

if __name__ == '__main__':
    main()
