#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_psemodel.py

Train YOLOv11-Fast using the pretrained “pse-mp46x/2” (COCOn) checkpoint
(mAP@50=92.4%, Precision=91.4%, Recall=78%) on 1200×800 images,
with your standard project-root layout:
  • data.yaml
  • train/images, train/labels
  • valid/images, valid/labels
  • test/images,  test/labels

Note: YOLOv11-Fast currently does not support built-in grayscale augmentation via API.
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    # ─── PARSE ARGS ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Train YOLOv11-Fast with pse-mp46x/2 pretrained weights'
    )
    parser.add_argument(
        '--root', '-r', type=str, required=True,
        help='Path to project root (must contain data.yaml, train/, valid/, test/)' 
    )
    parser.add_argument(
        '--model', '-m', type=str, default='pse-mp46x/2',
        help='Pretrained YOLOv11-Fast model URL or local path (e.g. pse-mp46x/2)'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', '-b', type=int, default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz', type=int, nargs=2, default=[1200, 800],
        metavar=('WIDTH', 'HEIGHT'),
        help='Training image size (width height)'
    )
    parser.add_argument(
        '--device', '-d', type=str, default='0',
        help='GPU device ID or "cpu"'
    )
    parser.add_argument(
        '--exp', '-n', type=str, default='yolov11_fast_pse',
        help='Experiment name (folder under runs/train/)'
    )
    args = parser.parse_args()

    # ─── CONFIG ──────────────────────────────────────────────────────────────────
    root      = os.path.abspath(args.root)
    data_yaml = os.path.join(root, 'data.yaml')

    splits = {
        'train': ('train/images', 'train/labels'),
        'valid': ('valid/images', 'valid/labels'),
        'test' : ('test/images',  'test/labels'),
    }

    # ─── SANITY CHECKS ────────────────────────────────────────────────────────────
    if not os.path.isfile(data_yaml):
        sys.exit(f"❌ data.yaml not found at {data_yaml}")
    for name, (img_sub, lbl_sub) in splits.items():
        img_dir = os.path.join(root, img_sub)
        lbl_dir = os.path.join(root, lbl_sub)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            sys.exit(f"❌ Missing '{name}' dirs:\n  {img_dir}\n  {lbl_dir}")
    # ──────────────────────────────────────────────────────────────────────────────

    # ─── AUGMENTATION SETTINGS ────────────────────────────────────────────────────
    AUG = dict(
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
    )
    # ──────────────────────────────────────────────────────────────────────────────

    print(f"🚀 Training YOLOv11-Fast on {root}")
    print(f" • data.yaml    : {data_yaml}")
    print(f" • pretrained   : {args.model} (COCOn checkpoint, mAP50=92.4%, P=91.4%, R=78%)")
    print(f" • epochs       : {args.epochs}")
    print(f" • batch size   : {args.batch}")
    print(f" • img size     : {tuple(args.imgsz)}")
    print(f" • device       : {args.device}")
    print(f" • experiment   : runs/train/{args.exp}\n")

    # ─── TRAIN ─────────────────────────────────────────────────────────────────────
    model = YOLO(args.model)  # loads pse-mp46x/2
    results = model.train(
        data        = data_yaml,
        epochs      = args.epochs,
        imgsz       = tuple(args.imgsz),
        batch       = args.batch,
        device      = args.device,
        project     = root,
        name        = args.exp,
        exist_ok    = False,
        save        = True,
        save_period = -1,
        augment     = True,
        **AUG
    )

    best = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"\n✅ Training complete! Best model saved at:\n   {best}")

if __name__ == '__main__':
    main()
