import os
import json
from glob import glob
from PIL import Image

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max

def convert_labelme_to_yolo(json_path, img_size):
    with open(json_path, 'r') as f:
        data = json.load(f)

    width, height = img_size
    yolo_lines = []

    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        if not points:
            continue
        x_min, y_min, x_max, y_max = polygon_to_bbox(points)

        # Normalize coordinates
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height

        # Assuming single class '0'
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_lines

def main():
    root_dir = os.getcwd()
    img_exts = ['.jpg', '.png', '.jpeg']
    image_files = []
    for ext in img_exts:
        image_files.extend(glob(os.path.join(root_dir, f'*{ext}')))

    os.makedirs('labels', exist_ok=True)

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(root_dir, base_name + '.json')
        if not os.path.exists(json_path):
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        yolo_lines = convert_labelme_to_yolo(json_path, (width, height))
        if yolo_lines:
            label_path = os.path.join(root_dir, 'labels', base_name + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

    # Create data.yaml
    data_yaml = f"""\
path: {root_dir}
train: {os.path.join(root_dir, 'images', 'train')}
val: {os.path.join(root_dir, 'images', 'val')}

nc: 1
names: ['object']
"""
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)

if __name__ == "__main__":
    main()