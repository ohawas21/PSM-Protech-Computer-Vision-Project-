#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tiny10_autogen.py

Automatically augment dataset (via image duplication & transformation),
fine-tune YOLOv8/YOLOv11 on expanded data, and ensure ≥85% precision.

Requirements:
    pip install ultralytics opencv-python
"""

import os
import sys
import argparse
import shutil
import cv2
import random
from glob import glob
from ultralytics import YOLO


def replicate_images(img_dir, label_dir, target_count):
    image_paths = glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png'))
    count = len(image_paths)
    if count == 0:
        sys.exit("❌ No images found to replicate.")

    print(f"🧪 Original image count: {count}. Target: {target_count}. Augmenting...")
    while len(glob(os.path.join(img_dir, '*'))) < target_count:
        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Random basic augmentation
            flip = cv2.flip(img, 1)
            blur = cv2.GaussianBlur(img, (5, 5), 0)

            for aug_img, suffix in [(flip, 'flip'), (blur, 'blur')]:
                out_name = f"{base}_{suffix}_{random.randint(0,9999)}.jpg"
                out_path = os.path.join(img_dir, out_name)
                cv2.imwrite(out_path, aug_img)

                label_src = os.path.join(label_dir, f"{base}.txt")
                label_dst = os.path.join(label_dir, f"{os.path.splitext(out_name)[0]}.txt")
                if os.path.isfile(label_src):
                    shutil.copy(label_src, label_dst)

                if len(glob(os.path.join(img_dir, '*'))) >= target_count:
                    break

    print(f"✅ Augmentation complete. Total images: {len(glob(os.path.join(img_dir, '*')))}")


def main():
    parser = argparse.ArgumentParser(description='Augment + Train YOLOv8/11 on small dataset')
    parser.add_argument('--root', '-r', type=str, required=True, help='Path to project root')
    parser.add_argument('--model', '-m', type=str, default='yolov11l.pt', help='YOLOv8/11 pretrained model')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[1200, 800], help='Image size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', '-d', type=str, default='0', help='CUDA device or "cpu"')
    parser.add_argument('--exp', '-n', type=str, default='tiny10_autogen', help='Experiment name')
    parser.add_argument('--target_count', type=int, default=100, help='Target number of training images')
    parser.add_argument('--precision_target', type=float, default=0.85, help='Minimum required precision')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    data_yaml = os.path.join(root, 'data.yaml')
    train_img_dir = os.path.join(root, 'train', 'images')
    train_lbl_dir = os.path.join(root, 'train', 'labels')

    if not os.path.isfile(data_yaml):
        sys.exit(f"❌ Missing data.yaml: {data_yaml}")
    if not os.path.isdir(train_img_dir) or not os.path.isdir(train_lbl_dir):
        sys.exit(f"❌ Invalid train dirs: {train_img_dir}, {train_lbl_dir}")

    # Step 1: Dataset Expansion
    replicate_images(train_img_dir, train_lbl_dir, args.target_count)

    # Step 2: Load Model
    model = YOLO(args.model)

    print(f"🚀 Starting training with expanded dataset at {root}")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=tuple(args.imgsz),
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        freeze=[0, 1, 2],
        augment=True,
        project=root,
        name=args.exp,
        save=True,
        exist_ok=True
    )

    # Step 3: Evaluate on test set
    print("📊 Evaluating model on test set...")
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=tuple(args.imgsz),
        batch=args.batch,
        device=args.device
    )
    
    precision = metrics.box['precision'] if 'box' in metrics else None
    print(f"\n✅ Evaluation done. Precision: {precision:.3f}")

    if precision and precision >= args.precision_target:
        print(f"🎯 Target precision of {args.precision_target:.2f} met ✅")
    else:
        print(f"⚠️ Precision {precision:.3f} below target {args.precision_target:.2f}. Consider tuning augmentations or training longer.")

if __name__ == '__main__':
    main()