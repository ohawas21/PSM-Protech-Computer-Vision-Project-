import shutil
import random
import yaml
import numpy as np
import time
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset")
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Folder containing images and .txt labels')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of data to use for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size (px) for training')
    parser.add_argument('--project_dir', type=str, default='runs',
                        help='YOLO project directory')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--model_save_path', type=str, required=True,
                        help='Full path to save the best model weights (.pt)')
    return parser.parse_args()

# Prepare dataset directories
def prepare_dirs(base: str = 'dataset') -> tuple[Path, Path]:
    """
    Creates train/val folders for images and labels.
    Returns (images_dir, labels_dir).
    """
    base_path = Path(base)
    img_path = base_path / 'images'
    lbl_path = base_path / 'labels'
    for split in ('train', 'val'):
        (img_path / split).mkdir(parents=True, exist_ok=True)
        (lbl_path / split).mkdir(parents=True, exist_ok=True)
    return img_path, lbl_path

# Split data and copy from a single source folder containing both images and .txt labels
def split_and_copy(
    src_dir: str,
    img_dir: Path,
    lbl_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> list[str]:
    """
    Splits images and their corresponding .txt label files from the same folder.
    Returns sorted list of class names inferred from filename prefixes.
    """
    src = Path(src_dir)
    images = list(src.glob('*.[jp][pn]g'))
    if not images:
        raise FileNotFoundError(f"No images found in {src_dir}")

    classes = sorted({img.stem.split('_')[0] for img in images})
    train_imgs, val_imgs = train_test_split(
        images, train_size=train_ratio, random_state=seed
    )
    splits = {'train': train_imgs, 'val': val_imgs}

    for split, img_list in splits.items():
        for img_path in img_list:
            # copy image
            (img_dir / split / img_path.name).write_bytes(img_path.read_bytes())
            # copy corresponding label if exists
            label_src = src / f"{img_path.stem}.txt"
            if label_src.exists():
                (lbl_dir / split / label_src.name).write_bytes(label_src.read_bytes())

    return classes

# Write YOLO YAML config file
def write_yaml(
    classes: list[str],
    img_dir: Path,
    out: str = 'data.yaml'
) -> str:
    cfg = {
        'train': str(img_dir / 'train'),
        'val': str(img_dir / 'val'),
        'nc': len(classes),
        'names': classes
    }
    Path(out).write_text(yaml.dump(cfg))
    return out

# Train and validate YOLO model
def train_yolo(
    config: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    project: str,
    name: str
) -> tuple[YOLO, any, Path]:
    # Choose device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():  # Apple MPS support
        device = 'mps'
    else:
        device = 'cpu'

    model = YOLO('yolov8n.pt')
    model.train(
        data=config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=3,
        seed=42
    )
    metrics = model.val()
    best_weights = Path(project) / name / 'weights' / 'best.pt'
    return model, metrics, best_weights

# Measure inference speed
def measure_speed(
    model: YOLO,
    img_dir: Path,
    runs: int = 50
) -> float:
    val_imgs = list(img_dir / 'val'.glob('*.[jp][pn]g'))[:runs]
    times = []
    for img_path in val_imgs:
        start = time.time()
        model.predict(str(img_path), conf=0.5, iou=0.75)
        times.append(time.time() - start)
    return np.mean(times) * 1000

# Main execution
def main():
    args = get_args()
    set_seed()

    # Prepare dataset structure
    img_dir, lbl_dir = prepare_dirs()
    classes = split_and_copy(
        src_dir=args.source_dir,
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        train_ratio=args.train_ratio
    )

    # Generate config
    cfg_path = write_yaml(classes, img_dir)

    # Train and save best model
    model, metrics, best_weights = train_yolo(
        config=cfg_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        project=args.project_dir,
        name=args.exp_name
    )
    if best_weights.exists():
        dest = Path(args.model_save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, dest)
        print(f"Best model saved to: {dest}")

    # Evaluate speed and save metrics
    speed_ms = measure_speed(model, img_dir)
    results = {
        'mAP50': metrics.results_dict['metrics/mAP50(B)'],
        'mAP50-95': metrics.results_dict['metrics/mAP50-95(B)'],
        'Speed_ms': speed_ms
    }
    results_path = Path(args.project_dir) / args.exp_name / 'results.yaml'
    results_path.write_text(yaml.dump(results))
    print("Results:", results)

if __name__ == '__main__':
    main()

'''
python train.py \
  --source_dir   data_source \
  --train_ratio  0.75 \
  --epochs       20 \
  --batch_size   16 \
  --img_size     640 \
  --project_dir  runs \
  --exp_name     my_experiment \
  --model_save_path  outputs/best_model.pt
  '''