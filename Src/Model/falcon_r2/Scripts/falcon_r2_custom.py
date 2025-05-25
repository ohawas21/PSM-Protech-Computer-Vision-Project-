import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def labelme_to_coco(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 0, "name": "object"}]
    ann_id = 1
    img_id = 1

    for json_file in tqdm(sorted(input_dir.rglob("*.json"))):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Warning] Skipping {json_file.name} - JSONDecodeError: {e}")
            continue

        image_path = input_dir / data.get("imagePath", "")
        if not image_path.exists():
            image_path = json_file.with_suffix(".jpg")  # fallback
            if not image_path.exists():
                continue

        try:
            image = Image.open(image_path)
        except Exception:
            continue

        width, height = image.size
        images.append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": image_path.name
        })

        for shape in data.get("shapes", []):
            if shape["shape_type"] != "polygon":
                continue
            points = shape["points"]
            segmentation = [coord for point in points for coord in point]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, y_min = min(x_coords), min(y_coords)
            width_box = max(x_coords) - x_min
            height_box = max(y_coords) - y_min

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "segmentation": [segmentation],
                "bbox": [x_min, y_min, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_dir / "train.json", 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"[INFO] COCO file saved at: {output_dir / 'train.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory with LabelMe JSONs and images")
    parser.add_argument("--output", default="./coco_output", help="Output directory to save COCO format")
    args = parser.parse_args()

    labelme_to_coco(args.input, args.output)