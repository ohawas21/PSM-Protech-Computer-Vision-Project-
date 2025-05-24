import os
import math
from PIL import Image, ImageDraw

def extract_regions(image_path: str, ann_path: str, output_dir: str) -> int:
    """
    Extracts YOLO bounding boxes and polygon regions from a blueprint image.

    - Bounding boxes: axis-aligned crops via PIL (no interpolation).
    - Polygons: masked, deskewed, and saved as PNG with alpha channel.
    """
    # Load image via PIL as RGBA
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    regions_count = 0
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            cls_id = parts[0]  # can be used to prefix output name if desired
            coords = list(map(float, parts[1:]))

            if len(coords) == 4:
                # YOLO bbox: x_center, y_center, width, height
                x_c, y_c, bw, bh = coords
                x1 = math.floor((x_c - bw/2) * w)
                y1 = math.floor((y_c - bh/2) * h)
                x2 = math.ceil((x_c + bw/2) * w)
                y2 = math.ceil((y_c + bh/2) * h)
                roi = img.crop((x1, y1, x2, y2))
            else:
                # Polygon: create mask and sub-image
                pts = [(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)]
                xs, ys = zip(*pts)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Crop sub-image and mask at polygon bounding box
                sub_img = img.crop((x_min, y_min, x_max, y_max))
                sub_w, sub_h = sub_img.size
                mask = Image.new('L', (sub_w, sub_h), 0)
                shifted_pts = [(x - x_min, y - y_min) for x, y in pts]
                ImageDraw.Draw(mask).polygon(shifted_pts, fill=255)

                # Determine deskew angle: minimal-area edge orientation
                def edge_angle(p0, p1):
                    return math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))

                angles = [abs(edge_angle(shifted_pts[i], shifted_pts[(i+1)%len(shifted_pts)])) % 180
                          for i in range(len(shifted_pts))]
                # Choose angle closest to 0 or 90
                best_angle = min(angles, key=lambda a: min(a, 180-a))
                # Adjust rotation to align
                rotate_angle = -best_angle if best_angle <= 90 else -(best_angle - 180)

                # Rotate sub-image and mask
                roi = sub_img.rotate(rotate_angle, expand=True, resample=Image.NEAREST)
                mask_rot = mask.rotate(rotate_angle, expand=True, resample=Image.NEAREST)

                # Ensure mask matches roi size
                if mask_rot.size != roi.size:
                    mask_rot = mask_rot.resize(roi.size, resample=Image.NEAREST)

                # Attach alpha channel
                roi.putalpha(mask_rot)

            # Save output
            filename = f"region_{regions_count:03d}.png"
            path_out = os.path.join(output_dir, filename)
            roi.save(path_out)
            regions_count += 1

    print(f"Extracted {regions_count} regions to '{output_dir}'")
    return regions_count


if __name__ == '__main__':
    IMAGE_PATH = r"PSM-Protech-Feasibility-Study\Src\Model\falcon_r1\dataset\test\images\230520_pdf_page_1_png.rf.e1e2bec49dc2a4800914d454dda9f896.jpg"
    ANN_PATH   = r"PSM-Protech-Feasibility-Study\Src\Model\falcon_r1\dataset\test\labels\230520_pdf_page_1_png.rf.e1e2bec49dc2a4800914d454dda9f896.txt"
    OUTPUT_DIR = r"PSM-Protech-Feasibility-Study\Documentation\testing"

    extract_regions(IMAGE_PATH, ANN_PATH, OUTPUT_DIR)