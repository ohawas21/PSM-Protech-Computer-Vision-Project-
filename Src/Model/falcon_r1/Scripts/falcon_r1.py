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
import json
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8 on tiny dataset (10 images) and evaluate'
    )
    parser.add_argument('--root', '-r', type=str, required=True,
                        help='Path to project root (data.yaml, train/, valid/, test/)')
    parser.add_argument('--model', '-m', type=str, default='yolov8l.pt',
                        help='Pretrained YOLOv8 weights')
    parser.add_argument('--epochs', '-e', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch', '-b', type=int, default=4,
                        help='Batch size (small for tiny dataset)')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[1200, 800],
                        metavar=('W', 'H'), help='Image size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--device', '-d', type=str, default='0',
                        help='CUDA device or "cpu"')
    parser.add_argument('--exp', '-n', type=str, default='tiny10_aug',
                        help='Experiment name')
    parser.add_argument('--metrics_out', type=str, default='metrics.json',
                        help='File to save evaluation metrics')
    args = parser.parse_args()

    # Paths
    root = os.path.abspath(args.root)
    data_yaml = os.path.join(root, 'data.yaml')
    for split in ['train', 'valid', 'test']:
        if not os.path.isdir(os.path.join(root, split, 'images')) or \
           not os.path.isdir(os.path.join(root, split, 'labels')):
            sys.exit(f"❌ Missing '{split}' images or labels folder")
    if not os.path.isfile(data_yaml):
        sys.exit(f"❌ data.yaml not found at {data_yaml}")

    # Load YOLOv8 model
    model = YOLO(args.model)

    # Aggressive augmentations
    aug = dict(
        mosaic=True, mixup=0.5, copy_paste=0.5,
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5
    )

    # Fine-tune
    print(f"🚀 Fine-tuning on tiny dataset at {root}")
    train_res = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=tuple(args.imgsz),
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        freeze=[0,1,2,3],
        augment=True,
        project=root,
        name=args.exp,
        exist_ok=False,
        **aug
    )
    best = os.path.join(train_res.save_dir, 'weights', 'best.pt')
    print(f"✅ Training complete. Best checkpoint: {best}\n")

    # Detect on test set
    test_src = os.path.join(root, 'test', 'images')
    print(f"🚀 Running detection on test set: {test_src}")
    _ = model.predict(
        source=test_src,
        imgsz=tuple(args.imgsz),
        conf=0.25,
        iou=0.45,
        device=args.device,
        save=True,
        project=root,
        name=f"{args.exp}_detect",
        exist_ok=True
    )

    # Evaluate on test set and capture metrics
    print("🚀 Evaluating on test set...")
    val_res = model.val(
        data=data_yaml,
        split='test',
        imgsz=tuple(args.imgsz),
        batch=args.batch,
        device=args.device
    )
    # val_res.metrics is a dict of metric names to values
    metrics = val_res.metrics if hasattr(val_res, 'metrics') else val_res[0].metrics
    print("✅ Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics to JSON
    out_path = os.path.join(root, args.metrics_out)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")

if __name__ == '__main__':
    main()
