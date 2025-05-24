#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tiny10.py

Fine-tune YOLOv8-L on a very small dataset (10 images) using aggressive augmentation
and transfer learning via freezing the backbone, then run detection and evaluation on the test set.

Usage:
    python3 fine_tiny10.py --root /path/to/project_root \
        [--model yolov8l.pt] [--epochs 30] [--batch 4] \
        [--imgsz 1200 800] [--lr 1e-4] [--device 0] [--exp tiny10_aug]

Requirements:
    pip install ultralytics

Your project_root must contain:
    data.yaml
    train/images/, train/labels/
    valid/images/, valid/labels/
    test/images/,  test/labels/
"""

import os
import sys
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8 on tiny dataset (10 images) and evaluate on test set'
    )
    parser.add_argument(
        '--root', '-r',
        type=str,
        required=True,
        help='Path to project root (data.yaml, train/, valid/, test/)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8l.pt',
        help='Pretrained YOLOv8 weights (e.g. yolov8l.pt)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=4,
        help='Batch size (small for tiny dataset)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        nargs=2,
        default=[1200, 800],
        metavar=('WIDTH', 'HEIGHT'),
        help='Training and inference image size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='0',
        help='CUDA device or "cpu"'
    )
    parser.add_argument(
        '--exp', '-n',
        type=str,
        default='tiny10_aug',
        help='Experiment name (runs/train/<exp>)'
    )
    args = parser.parse_args()

    # Resolve paths
    root = os.path.abspath(args.root)
    data_yaml = os.path.join(root, 'data.yaml')

    # Sanity checks
    if not os.path.isfile(data_yaml):
        sys.exit(f"❌ data.yaml not found at {data_yaml}")
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(root, split, 'images')
        lbl_dir = os.path.join(root, split, 'labels')
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            sys.exit(f"❌ Missing folders for '{split}': {img_dir}, {lbl_dir}")

    # Load model
    model = YOLO(args.model)

    # Aggressive augmentation & transfer learning settings
    augment_kwargs = dict(
        mosaic      = True,
        mixup       = 0.5,
        copy_paste  = 0.5,
        hsv_h       = 0.02,
        hsv_s       = 0.8,
        hsv_v       = 0.5,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.5,
        fliplr      = 0.5,
    )

    print(f"🚀 Fine-tuning on tiny dataset (10 images) at: {root}")
    print(f" • data.yaml   : {data_yaml}")
    print(f" • model       : {args.model}")
    print(f" • epochs      : {args.epochs}")
    print(f" • batch size  : {args.batch}")
    print(f" • img size    : {tuple(args.imgsz)}")
    print(f" • lr          : {args.lr}")
    print(f" • device      : {args.device}")
    print(f" • experiment  : runs/train/{args.exp}\n")

    # Train with freezing backbone
    train_results = model.train(
        data          = data_yaml,
        epochs        = args.epochs,
        imgsz         = tuple(args.imgsz),
        batch         = args.batch,
        lr0           = args.lr,
        device        = args.device,
        freeze        = [0, 1, 2, 3],  # freeze first modules
        augment       = True,
        project       = root,
        name          = args.exp,
        save          = True,
        exist_ok      = False,
        **augment_kwargs
    )

    best_ckpt = os.path.join(train_results.save_dir, 'weights', 'best.pt')
    print(f"\n✅ Fine-tuning complete! Best model saved at:\n   {best_ckpt}\n")

    # Run detection on test set
    test_source = os.path.join(root, 'test', 'images')
    print(f"🚀 Running detection on test images: {test_source}")
    _ = model.predict(
        source      = test_source,
        imgsz       = tuple(args.imgsz),
        conf        = 0.25,
        iou         = 0.45,
        device      = args.device,
        save        = True,
        project     = root,
        name        = f"{args.exp}_test_detect",
        exist_ok    = True
    )
    print(f"✅ Detection images saved under runs/detect/{args.exp}_test_detect/\n")

    # Evaluate on test set
    print("🚀 Evaluating on test set...")
    val_results = model.val(
        data        = data_yaml,
        split       = 'test',
        imgsz       = tuple(args.imgsz),
        batch       = args.batch,
        device      = args.device
    )
    # The .val() method prints metrics; optionally capture them as dict
    print("✅ Evaluation complete!")

if __name__ == '__main__':
    main()
