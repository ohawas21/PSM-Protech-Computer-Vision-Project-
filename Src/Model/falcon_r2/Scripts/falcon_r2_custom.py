import os
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import argparse
import random
import shutil

# 1. Convert LabelMe JSON to YOLO Segmentation Format
def convert_polygon_to_yolo(points, img_w, img_h):
    return [coord / img_w if i % 2 == 0 else coord / img_h for i, coord in enumerate(sum(points, []))]

def convert_labelme_dataset(labelme_dir, yolo_label_dir, class_map):
    os.makedirs(yolo_label_dir, exist_ok=True)

    for json_file in Path(labelme_dir).rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] Skipping corrupt JSON during conversion: {json_file.name}")
            continue

        image_path = Path(labelme_dir, data["imagePath"])
        if not image_path.exists():
            print(f"[WARN] Image not found for {json_file.name}, skipping.")
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        output_lines = []
        for shape in data["shapes"]:
            if shape["shape_type"] != "polygon":
                continue

            class_id = class_map.get(shape["label"])
            if class_id is None:
                print(f"[WARN] Unknown label: {shape['label']}")
                continue

            norm_coords = convert_polygon_to_yolo(shape["points"], img_w, img_h)
            output_lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_coords))

        with open(Path(yolo_label_dir, json_file.stem + ".txt"), "w") as f:
            f.write("\n".join(output_lines))

# 2. Generate data.yaml
def generate_data_yaml(save_path, class_map, train_img, val_img):
    with open(save_path, "w") as f:
        f.write(f"path: {Path(train_img).parent.parent.resolve()}\n")
        f.write(f"train: {Path(train_img).parent.as_posix()}\n")
        f.write(f"val: {Path(val_img).parent.as_posix()}\n")
        f.write("names:\n")
        for id, name in class_map.items():
            f.write(f"  {id}: '{name}'\n")

# 3. Auto pipeline
def auto_train_pipeline(labelme_train, labelme_val, dataset_root, class_list, model_path='yolov8l-seg.pt'):
    class_map = {label: idx for idx, label in enumerate(class_list)}
    structure = ['train', 'val']

    for split in structure:
        img_dir = Path(dataset_root, split, "images")
        lbl_dir = Path(dataset_root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        src_json_dir = Path(labelme_train if split == "train" else labelme_val)
        for json_file in Path(src_json_dir).rglob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"[ERROR] Skipping corrupt JSON: {json_file.name}")
                continue
            image_file = Path(src_json_dir, data["imagePath"])
            if image_file.exists():
                os.system(f'cp "{image_file}" "{img_dir}/{image_file.name}"')  # copy images

        convert_labelme_dataset(src_json_dir, lbl_dir, class_map)

    yaml_path = Path(dataset_root, "data.yaml")
    sample_train_img = next(Path(dataset_root, "train", "images").glob("*.png"))
    sample_val_img = next(Path(dataset_root, "val", "images").glob("*.png"))
    generate_data_yaml(yaml_path, {v: k for k, v in class_map.items()}, sample_train_img, sample_val_img)

    # 4. Train YOLOv8
    model = YOLO(model_path)
    model.train(
        data=str(yaml_path),
        epochs=300,
        imgsz=640,
        task="segment",
        name="auto_seg_train"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation model from LabelMe data")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing LabelMe JSON and images")
    args = parser.parse_args()

    root_path = Path(args.root)
    all_jsons = list(root_path.rglob("*.json"))
    random.seed(42)
    random.shuffle(all_jsons)

    split_idx = int(0.8 * len(all_jsons))
    train_jsons = all_jsons[:split_idx]
    val_jsons = all_jsons[split_idx:]

    tmp_root = Path("Dataset_split")
    labelme_train = tmp_root / "labelme_train"
    labelme_val = tmp_root / "labelme_val"
    os.makedirs(labelme_train, exist_ok=True)
    os.makedirs(labelme_val, exist_ok=True)

    for f in train_jsons:
        shutil.copy(f, labelme_train / f.name)
        try:
            with open(f) as jf:
                data = json.load(jf)
            img_file = f.parent / data["imagePath"]
        except json.JSONDecodeError:
            print(f"[ERROR] Skipping corrupt JSON: {f.name}")
            continue
        if img_file.exists():
            shutil.copy(img_file, labelme_train / img_file.name)

    for f in val_jsons:
        shutil.copy(f, labelme_val / f.name)
        try:
            with open(f) as jf:
                data = json.load(jf)
            img_file = f.parent / data["imagePath"]
        except json.JSONDecodeError:
            print(f"[ERROR] Skipping corrupt JSON: {f.name}")
            continue
        if img_file.exists():
            shutil.copy(img_file, labelme_val / img_file.name)

    auto_train_pipeline(
        labelme_train=labelme_train,
        labelme_val=labelme_val,
        dataset_root="Dataset",  # will structure Dataset/train/images etc.
        class_list=["falcon_r2"],  # can be extended
        model_path="yolov8l-seg.pt"
    )