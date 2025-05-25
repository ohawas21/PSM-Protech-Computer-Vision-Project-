import os
import json
import argparse
import shutil
from glob import glob
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def labelme_to_coco(root_dir, output_dir, splits=(0.7, 0.2, 0.1)):
    os.makedirs(output_dir, exist_ok=True)
    jsons = sorted(glob(os.path.join(root_dir, '*.json')))
    imgs = sorted([f for f in glob(os.path.join(root_dir, '*')) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    # Shuffle
    data = list(zip(jsons, imgs))
    random.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * splits[0])
    n_val = int(n_total * splits[1])

    split_sets = {
        'train': data[:n_train],
        'val': data[n_train:n_train + n_val],
        'test': data[n_train + n_val:]
    }

    categories = [{"id": 0, "name": "object"}]  # Edit for multiple classes

    for split, items in split_sets.items():
        ann = {"images": [], "annotations": [], "categories": categories}
        split_img_dir = os.path.join(output_dir, split, "images")
        os.makedirs(split_img_dir, exist_ok=True)
        img_id = 0
        ann_id = 0

        for json_path, img_path in items:
            with open(json_path, 'r') as f:
                data = json.load(f)

            img = Image.open(img_path)
            w, h = img.size
            fname = os.path.basename(img_path)
            new_img_path = os.path.join(split_img_dir, fname)
            shutil.copy(img_path, new_img_path)

            ann['images'].append({
                "id": img_id,
                "file_name": fname,
                "width": w,
                "height": h
            })

            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "polygon":
                    continue
                points = shape["points"]
                segmentation = [coord for pt in points for coord in pt]
                x_coords = [pt[0] for pt in points]
                y_coords = [pt[1] for pt in points]
                bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]

                ann["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [segmentation],
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0
                })
                ann_id += 1
            img_id += 1

        with open(os.path.join(output_dir, f"annotations_{split}.json"), 'w') as f:
            json.dump(ann, f)

    print("COCO dataset generated at:", output_dir)

def create_data_yaml(output_path):
    content = f"""\
path: {output_path}
train: images/train
val: images/val
test: images/test

coco_train_json: annotations_train.json
coco_val_json: annotations_val.json
coco_test_json: annotations_test.json

names:
  0: object
"""
    with open(os.path.join(output_path, "data.yaml"), "w") as f:
        f.write(content)

def train_yolo(data_yaml_path, model="yolov8l-seg.pt", epochs=300):
    model = YOLO(model)
    model.train(data=data_yaml_path, epochs=epochs, imgsz=640, task="segment", format="coco")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to LabelMe images + JSONs")
    parser.add_argument("--output", type=str, default="./YOLO_COCO", help="Path to output directory")
    args = parser.parse_args()

    # Convert
    labelme_to_coco(args.input, args.output)
    create_data_yaml(args.output)

    # Train
    train_yolo(os.path.join(args.output, "data.yaml"))