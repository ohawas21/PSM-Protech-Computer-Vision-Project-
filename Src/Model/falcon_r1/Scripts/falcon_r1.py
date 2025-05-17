#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v8_with_roboflow.py

Train YOLOv8-L on either a local project root or a Roboflow-hosted dataset.
Supports v8-style augmentations (mosaic, mixup, copy-paste, HSV jitters, flips).

Usage examples:
  # Local dataset:
  python train_v8_with_roboflow.py \
    --root /path/to/project_root \
    --model yolov8l.pt \
    --epochs 50 --batch 8 --imgsz 1200 800 \
    --device 0 --exp yolov8_rf_local

  # Roboflow dataset:
  python train_v8_with_roboflow.py \
    --roboflow_key YOUR_API_KEY \
    --rf_workspace WORKSPACE_NAME \
    --rf_project PROJECT_NAME \
    --rf_version VERSION_NUMBER \
    --model yolov8l.pt \
    --epochs 50 --batch 8 --imgsz 1200 800 \
    --device 0 --exp yolov8_rf
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8-L locally or via Roboflow'
    )
    root_group = parser.add_mutually_exclusive_group(required=False)
    root_group.add_argument('--root', '-r', type=str,
                            help='Local project root (data.yaml, train/, valid/, test/)')
    root_group.add_argument('--roboflow_key', type=str,
                            help='Roboflow API key to download dataset')
    parser.add_argument('--rf_workspace', type=str, default=None,
                        help='Roboflow workspace name')
    parser.add_argument('--rf_project', type=str, default=None,
                        help='Roboflow project name')
    parser.add_argument('--rf_version', type=int, default=None,
                        help='Roboflow version number')
    parser.add_argument('--model', '-m', type=str, default='yolov8l.pt',
                        help='YOLOv8 weights (e.g. yolov8l.pt)')
    parser.add_argument('--epochs','-e', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', '-b', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[1200,800],
                        metavar=('WIDTH','HEIGHT'),
                        help='Image size: width height')
    parser.add_argument('--device', '-d', type=str, default='0',
                        help='GPU device or "cpu"')
    parser.add_argument('--exp', '-n', type=str, default='yolov8_rf',
                        help='Experiment name (runs/train/<exp>)')
    args = parser.parse_args()

    # Determine data.yaml and root
    if args.roboflow_key:
        try:
            from roboflow import Roboflow
        except ImportError:
            sys.exit("❌ Install roboflow: pip install roboflow")
        rf = Roboflow(api_key=args.roboflow_key)
        workspace = rf.workspace(args.rf_workspace) if args.rf_workspace else rf.workspace()
        project   = workspace.project(args.rf_project)
        version   = project.version(args.rf_version)
        dataset   = version.download("yolov8")
        data_yaml = os.path.join(dataset.location, 'data.yaml')
        root      = dataset.location
    else:
        if not args.root:
            sys.exit("❌ Either --root or --roboflow_key must be provided.")
        root      = os.path.abspath(args.root)
        data_yaml = os.path.join(root, 'data.yaml')
        # local sanity checks
        splits = {'train':('train/images','train/labels'),
                  'valid':('valid/images','valid/labels'),
                  'test': ('test/images','test/labels')}
        if not os.path.isfile(data_yaml):
            sys.exit(f"❌ data.yaml not found at {data_yaml}")
        for name,(imgs,lbls) in splits.items():
            img_dir = os.path.join(root, imgs)
            lbl_dir = os.path.join(root, lbls)
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                sys.exit(f"❌ Missing '{name}' dirs: {img_dir}, {lbl_dir}")

    # v8 augmentation parameters
    AUG = dict(
        mosaic      = True,
        mixup       = 0.15,
        copy_paste  = 0.10,
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

    print(f"🚀 Training on dataset at: {root}")
    print(f" • data.yaml  : {data_yaml}")
    print(f" • weights    : {args.model}")
    print(f" • epochs     : {args.epochs}")
    print(f" • batch size : {args.batch}")
    print(f" • img size   : {tuple(args.imgsz)}")
    print(f" • device     : {args.device}")
    print(f" • exp name   : runs/train/{args.exp}\n")

    # Load YOLOv8-L and train
    model = YOLO(args.model)
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
    print(f"\n✅ Training complete! Best model at: {best}")

if __name__ == '__main__':
    main()
