#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train YOLOv11-L object detector with grayscale augmentation (15%).
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv11-L with 15% grayscale augmentation'
    )
    parser.add_argument('--root',  '-r', type=str, required=True,
                        help='Project root (data.yaml, train/, valid/, test/)')
    parser.add_argument('--model', '-m', type=str, default='yolov11l.pt',
                        help='Weights file (e.g. yolov11l.pt)')
    parser.add_argument('--epochs','-e', type=int, default=50)
    parser.add_argument('--batch', '-b', type=int, default=8)
    parser.add_argument('--imgsz',          type=int, nargs=2,
                        default=[1200,800], metavar=('W','H'))
    parser.add_argument('--device','-d',    type=str, default='0')
    parser.add_argument('--exp',   '-n',    type=str,
                        default='yolov11l_gray15',
                        help='runs/train/<exp_name>')
    args = parser.parse_args()

    # Paths
    root = os.path.abspath(args.root)
    data_yaml = os.path.join(root, 'data.yaml')
    splits = {
        'train': ('train/images',  'train/labels'),
        'valid': ('valid/images',  'valid/labels'),
        'test':  ('test/images',   'test/labels'),
    }
    # sanity checks
    if not os.path.isfile(data_yaml):
        sys.exit(f"❌ data.yaml missing at {data_yaml}")
    for name, (img_sub, lbl_sub) in splits.items():
        img_dir = os.path.join(root, img_sub)
        lbl_dir = os.path.join(root, lbl_sub)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            sys.exit(f"❌ Missing {name} folder(s): {img_dir}, {lbl_dir}")

    # Hyperparams & augmentations
    model = YOLO(args.model)
    aug = dict(
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
        grayscale   = 0.15,  # <-- only works in Python API
    )

    print(f"🚀 Training on {root} with 15% grayscale augment…")
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
        **aug
    )

    best = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"\n✅ Done! Best model at {best}")

if __name__ == '__main__':
    main()
