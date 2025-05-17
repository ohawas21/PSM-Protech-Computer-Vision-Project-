#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train YOLOv11-L object detector with grayscale augmentation (15%)
on 1200×800 images, using an arbitrary project root folder.
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    # ─── PARSE ARGS ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Train YOLOv11-L with grayscale augmentation (15%)'
    )
    parser.add_argument(
        '--root', '-r',
        type=str,
        required=True,
        help='Path to your project root (must contain data.yaml, train/, valid/, test/)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov11l.pt',
        help='Pretrained weights or checkpoint'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        nargs=2,
        default=[1200, 800],
        metavar=('WIDTH', 'HEIGHT'),
        help='Training image size (width height)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='0',
        help='GPU device (e.g. "0") or "cpu"'
    )
    parser.add_argument(
        '--exp', '-n',
        type=str,
        default='yolov11l_gray15',
        help='Experiment name (folder under runs/train/)'
    )
    args = parser.parse_args()

    # ─── CONFIG ──────────────────────────────────────────────────────────────────
    PROJECT_ROOT = os.path.abspath(args.root)
    DATA_YAML    = os.path.join(PROJECT_ROOT, 'data.yaml')

    # Folders for splits
    TRAIN_IMAGES  = os.path.join(PROJECT_ROOT, 'train', 'images')
    TRAIN_LABELS  = os.path.join(PROJECT_ROOT, 'train', 'labels')
    VALID_IMAGES  = os.path.join(PROJECT_ROOT, 'valid', 'images')
    VALID_LABELS  = os.path.join(PROJECT_ROOT, 'valid', 'labels')
    TEST_IMAGES   = os.path.join(PROJECT_ROOT, 'test',  'images')
    TEST_LABELS   = os.path.join(PROJECT_ROOT, 'test',  'labels')

    MODEL_NAME   = args.model
    EPOCHS       = args.epochs
    BATCH_SIZE   = args.batch
    IMGSZ        = tuple(args.imgsz)
    DEVICE       = args.device
    EXP_NAME     = args.exp

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

    # ─── SANITY CHECKS ────────────────────────────────────────────────────────────
    if not os.path.isfile(DATA_YAML):
        sys.exit(f"❌ data.yaml not found at {DATA_YAML}")

    for split in ('train', 'valid', 'test'):
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

    # Train with augmentations (expand AUG_KWARGS with **)
    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH_SIZE,
        device      = DEVICE,
        project     = PROJECT_ROOT,
        name        = EXP_NAME,
        exist_ok    = False,
        save        = True,
        save_period = -1,
        augment     = True,
        **AUG_KWARGS
    )

    # Report best checkpoint
    best_ckpt = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"\n✅ Training complete! Best model: {best_ckpt}")

if __name__ == '__main__':
    main()
