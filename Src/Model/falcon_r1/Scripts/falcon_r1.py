#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
falcon_r1_v8.py

Train YOLOv8-L with typical v8 augmentations on 1200×800 images,
using a project root that contains:
  • data.yaml
  • train/images, train/labels
  • valid/images, valid/labels
  • test/images,  test/labels
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    # ─── PARSE ARGS ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Train YOLOv8-L with v8-style augmentations'
    )
    parser.add_argument('--root',  '-r', type=str, required=True,
                        help='Path to project root (data.yaml, train/, valid/, test/)')
    parser.add_argument('--model', '-m', type=str, default='yolov8l.pt',
                        help='YOLOv8 weights file (e.g. yolov8l.pt)')
    parser.add_argument('--epochs','-e', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', '-b', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz',          type=int, nargs=2,
                        default=[1200,800], metavar=('WIDTH','HEIGHT'),
                        help='Training image size: width height')
    parser.add_argument('--device','-d',    type=str, default='0',
                        help='GPU device ID or "cpu"')
    parser.add_argument('--exp',   '-n',    type=str,
                        default='yolov8l_aug_v1',
                        help='Experiment name (folder under runs/train/)')
    args = parser.parse_args()

    # ─── SETUP ─────────────────────────────────────────────────────────────────────
    root      = os.path.abspath(args.root)
    data_yaml = os.path.join(root, 'data.yaml')
    splits = {
        'train': ('train/images', 'train/labels'),
        'valid': ('valid/images', 'valid/labels'),
        'test' : ('test/images',  'test/labels'),
    }

    # Sanity checks
    if not os.path.isfile(data_yaml):
        sys.exit(f"❌ data.yaml not found at {data_yaml}")
    for name, (imgs, lbls) in splits.items():
        img_dir = os.path.join(root, imgs)
        lbl_dir = os.path.join(root, lbls)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            sys.exit(f"❌ Missing '{name}' dirs:\n  {img_dir}\n  {lbl_dir}")

    # ─── AUGMENTATION KWARGS ───────────────────────────────────────────────────────
    AUG = dict(
        mosaic        = True,    # enable Mosaic
        mixup         = 0.15,    # mixup probability
        copy_paste    = 0.10,    # copy-paste probability
        hsv_h         = 0.015,   # HSV hue jitter
        hsv_s         = 0.7,     # HSV saturation jitter
        hsv_v         = 0.4,     # HSV value jitter
        degrees       = 2.0,     # rotation
        translate     = 0.08,    # translation
        scale         = 0.5,     # scale jitter
        shear         = 0.0,     # shear
        perspective   = 0.0,     # perspective
        flipud        = 0.0,     # vertical flip
        fliplr        = 0.5,     # horizontal flip
    )
    # ──────────────────────────────────────────────────────────────────────────────

    print(f"🚀 Starting YOLOv8-L training on {root}")
    print(f" • data.yaml  : {data_yaml}")
    print(f" • weights    : {args.model}")
    print(f" • epochs     : {args.epochs}")
    print(f" • batch size : {args.batch}")
    print(f" • img size   : {tuple(args.imgsz)}")
    print(f" • device     : {args.device}")
    print(f" • exp name   : runs/train/{args.exp}\n")

    # ─── TRAIN ─────────────────────────────────────────────────────────────────────
    model = YOLO(args.model)  # load YOLOv8-L
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
    print(f"\n✅ Training finished! Best model at:\n   {best}")

if __name__ == '__main__':
    main()
