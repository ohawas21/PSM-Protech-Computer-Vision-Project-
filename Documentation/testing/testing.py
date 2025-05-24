import os
import math
import logging
from typing import Tuple
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def unsharp_mask(img: Image.Image, sigma: float = 1.0, amount: float = 1.5) -> Image.Image:
    """Apply unsharp mask to sharpen the image."""
    arr = np.array(img.convert('L'), dtype=np.float32)
    blurred = gaussian_filter(arr, sigma=sigma)
    sharp = arr + amount * (arr - blurred)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp).convert('RGBA')

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """Clamp bounding box to image dimensions."""
    x1c = max(0, min(x1, w-1))
    y1c = max(0, min(y1, h-1))
    x2c = max(1, min(x2, w))
    y2c = max(1, min(y2, h))
    return x1c, y1c, x2c, y2c

def extract_regions(image_path: str,
                    ann_path:     str,
                    output_dir:   str,
                    apply_sharpen: bool = True) -> int:
    """
    - Optionally sharpen the source image.
    - Extract axis-aligned YOLO crops or polygon regions (masked + deskewed).
    - Save each as `<classID>_region_###.png` in output_dir.
    """
    img = Image.open(image_path).convert('RGBA')
    w, h = img.size

    if apply_sharpen:
        logging.info("Sharpening source image with Unsharp Mask…")
        img = unsharp_mask(img)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            # === YOLO bbox (4 coords) ===
            if len(coords) == 4:
                x_c, y_c, bw, bh = coords
                x1 = math.floor((x_c - bw/2) * w)
                y1 = math.floor((y_c - bh/2) * h)
                x2 = math.ceil ((x_c + bw/2) * w)
                y2 = math.ceil ((y_c + bh/2) * h)
                x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
                roi = img.crop((x1, y1, x2, y2))

            # === Polygon (>4 coords) ===
            else:
                pts = [(coords[i]*w, coords[i+1]*h)
                       for i in range(0, len(coords), 2)]
                xs, ys = zip(*pts)
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                x_min, y_min, x_max, y_max = clamp_bbox(x_min, y_min, x_max, y_max, w, h)

                sub = img.crop((x_min, y_min, x_max, y_max))
                sw, sh = sub.size
                mask = Image.new('L', (sw, sh), 0)
                shifted = [(x - x_min, y - y_min) for x, y in pts]
                ImageDraw.Draw(mask).polygon(shifted, fill=255)

                # Deskew: align longest edge to 0°/90°
                def edge_ang(p0, p1):
                    return math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
                # find longest edge
                lengths = [
                    (math.hypot(shifted[i][0]-shifted[(i+1)%len(shifted)][0],
                                shifted[i][1]-shifted[(i+1)%len(shifted)][1]),
                     i)
                    for i in range(len(shifted))
                ]
                _, idx = max(lengths)
                raw = edge_ang(shifted[idx], shifted[(idx+1)%len(shifted)])
                snapped = round(raw / 90) * 90
                angle = snapped - raw

                roi = sub.rotate(angle, expand=True, resample=Image.NEAREST)
                m2  = mask.rotate(angle, expand=True, resample=Image.NEAREST)
                if m2.size != roi.size:
                    m2 = m2.resize(roi.size, resample=Image.NEAREST)
                roi.putalpha(m2)

            out_name = f"{cls_id}_region_{count:03d}.png"
            out_path = os.path.join(output_dir, out_name)
            roi.save(out_path)
            logging.info(f"→ Saved {out_name}")
            count += 1

    logging.info(f"Done: extracted {count} regions into “{output_dir}”")
    return count

if __name__ == '__main__':
    IMAGE_PATH = r"C:/Users/alyan/PSE/PSM-Protech-Feasibility-Study/Src/Model/falcon_r1/dataset/test/images/230520_pdf_page_1_png.rf.e1e2bec49dc2a4800914d454dda9f896.jpg"
    ANN_PATH   = r"C:/Users/alyan/PSE/PSM-Protech-Feasibility-Study/Src/Model/falcon_r1/dataset/test/labels/230520_pdf_page_1_png.rf.e1e2bec49dc2a4800914d454dda9f896.txt"
    OUTPUT_DIR = r"C:/Users/alyan/PSE/PSM-Protech-Feasibility-Study/Documentation/testing"

    extract_regions(IMAGE_PATH, ANN_PATH, OUTPUT_DIR, apply_sharpen=True)
