# yolo_labelme_trainer.py
"""
YOLO Training Script for LabelMe Datasets
=========================================
Simple, robust YOLO training with automatic LabelMe JSON to YOLO format conversion.

Features:
- Automatic LabelMe polygon → YOLO bbox conversion
- Graceful handling of missing annotations
- Auto train/val split
- Crop extraction after training
- GPU acceleration when available

Install:
-------
pip install ultralytics pillow

Usage:
------
python yolo_labelme_trainer.py --data_dir ../Dataset --img_exts png
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import yaml

from PIL import Image
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("yolo_trainer")

def labelme_to_yolo_bbox(points: List[List[float]], img_width: int, img_height: int) -> List[float]:
    """Convert LabelMe polygon points to YOLO format bbox (normalized center_x, center_y, width, height)"""
    xs, ys = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Convert to YOLO format (normalized)
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [center_x, center_y, width, height]

def process_labelme_json(json_path: Path, img_width: int, img_height: int) -> Tuple[List[List[float]], List[str]]:
    """Extract bounding boxes and labels from LabelMe JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        log.warning(f"Could not read {json_path}: {e}")
        return [], []
    
    bboxes = []
    labels = []
    
    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        label = shape.get('label', 'unknown')
        
        if len(points) < 2:
            continue
            
        try:
            bbox = labelme_to_yolo_bbox(points, img_width, img_height)
            bboxes.append(bbox)
            labels.append(label)
        except Exception as e:
            log.warning(f"Error processing shape in {json_path}: {e}")
            continue
    
    return bboxes, labels

def find_annotation_file(img_path: Path) -> Path:
    """Find corresponding JSON annotation file for an image"""
    # Try different naming conventions
    candidates = [
        img_path.with_suffix('.json'),  # img.png -> img.json
        img_path.with_suffix(img_path.suffix + '.json'),  # img.png -> img.png.json
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(f"No annotation found for {img_path.name}")

def convert_dataset_to_yolo(data_dir: Path, output_dir: Path, img_extensions: List[str]):
    """Convert LabelMe dataset to YOLO format"""
    log.info("Converting LabelMe dataset to YOLO format...")
    
    # Find all images
    image_files = []
    for ext in img_extensions:
        image_files.extend(data_dir.glob(f"*.{ext.lstrip('.')}"))
    
    if not image_files:
        raise FileNotFoundError(f"No images found with extensions {img_extensions} in {data_dir}")
    
    log.info(f"Found {len(image_files)} images")
    
    # Create output directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all valid images with annotations
    valid_files = []
    all_labels = set()
    skipped_count = 0
    
    for img_path in image_files:
        try:
            json_path = find_annotation_file(img_path)
            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            bboxes, labels = process_labelme_json(json_path, img_width, img_height)
            
            if not bboxes:
                log.warning(f"Skip {img_path.name}: no valid annotations")
                skipped_count += 1
                continue
            
            valid_files.append((img_path, json_path, bboxes, labels))
            all_labels.update(labels)
            
        except Exception as e:
            log.warning(f"Skip {img_path.name}: {e}")
            skipped_count += 1
            continue
    
    if not valid_files:
        raise RuntimeError("No valid annotated images found!")
    
    log.info(f"Valid images: {len(valid_files)}, Skipped: {skipped_count}")
    log.info(f"Found classes: {sorted(all_labels)}")
    
    # Create class mapping
    class_names = sorted(all_labels)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Split train/val (80/20)
    split_idx = int(0.8 * len(valid_files))
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    log.info(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Process train files
    for img_path, json_path, bboxes, labels in train_files:
        # Copy image
        shutil.copy2(img_path, train_img_dir / img_path.name)
        
        # Create YOLO label file
        label_file = train_label_dir / f"{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            for bbox, label in zip(bboxes, labels):
                class_id = class_to_id[label]
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
    
    # Process val files
    for img_path, json_path, bboxes, labels in val_files:
        # Copy image
        shutil.copy2(img_path, val_img_dir / img_path.name)
        
        # Create YOLO label file
        label_file = val_label_dir / f"{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            for bbox, label in zip(bboxes, labels):
                class_id = class_to_id[label]
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    log.info(f"Dataset converted successfully!")
    log.info(f"Dataset config saved to: {yaml_path}")
    
    return yaml_path, class_names

def train_yolo_model(dataset_yaml: Path, epochs: int, img_size: int, batch_size: int):
    """Train YOLO model"""
    log.info("Starting YOLO training...")
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, etc.
    
    # Train the model
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        save=True,
        device='0' if hasattr(model, 'device') else 'cpu'  # Use GPU if available
    )
    
    log.info("Training completed!")
    return model

def extract_crops(model, data_dir: Path, output_dir: Path, conf_threshold: float = 0.5):
    """Extract crops from trained model predictions"""
    log.info("Extracting crops from predictions...")
    
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    
    # Find all images in original dataset
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(data_dir.glob(f"*.{ext}"))
    
    crop_count = 0
    for img_path in image_files:
        try:
            # Run inference
            results = model(str(img_path), conf=conf_threshold)
            
            # Extract crops
            for i, result in enumerate(results):
                if len(result.boxes) > 0:
                    # Save crops
                    crops = result.save_crop(
                        save_dir=crop_dir,
                        file_name=f"{img_path.stem}_crop"
                    )
                    crop_count += len(result.boxes)
                    
        except Exception as e:
            log.warning(f"Error processing {img_path.name}: {e}")
            continue
    
    log.info(f"Extracted {crop_count} crops to {crop_dir}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Training for LabelMe Datasets")
    parser.add_argument("--data_dir", required=True, help="Directory containing images and JSON annotations")
    parser.add_argument("--output_dir", default="./yolo_dataset", help="Output directory for converted dataset")
    parser.add_argument("--crops_dir", default="./crops", help="Directory to save extracted crops")
    parser.add_argument("--img_exts", default="jpg,jpeg,png", help="Image extensions (comma-separated)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=640, help="Training image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for crop extraction")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    crops_dir = Path(args.crops_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Parse image extensions
    img_extensions = [ext.strip().lstrip('.') for ext in args.img_exts.split(',')]
    
    try:
        # Step 1: Convert dataset to YOLO format
        dataset_yaml, class_names = convert_dataset_to_yolo(data_dir, output_dir, img_extensions)
        
        # Step 2: Train YOLO model
        model = train_yolo_model(dataset_yaml, args.epochs, args.img_size, args.batch_size)
        
        # Step 3: Extract crops
        extract_crops(model, data_dir, crops_dir, args.conf_threshold)
        
        log.info("All steps completed successfully!")
        log.info(f"Model weights saved in: runs/detect/train/weights/best.pt")
        log.info(f"Crops saved in: {crops_dir}")
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()