import os
import json
import glob
import cv2
import numpy as np

# === 1) Compute absolute paths based on script location ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",  # up from Scripts
        "Dataset",
        "prean"
    )
)

OUTPUT_DIR = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",  # up from Scripts
        "bigcrop_output"
    )
)

print(f"[INFO] INPUT_DIR:  {INPUT_DIR}")
print(f"[INFO] OUTPUT_DIR: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2) Gather all images (recursively) ===
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

def get_all_image_paths(root_dir):
    img_paths = []
    for ext in IMG_EXTS:
        # glob.glob with recursive
        img_paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return img_paths

# === 3) JSON loader ===
def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

# === 4) Drawing helper ===
def draw_polygons_on_image(img, shapes, color=(0,255,0), thickness=2, fill=False):
    overlay = img.copy()
    for shape in shapes:
        pts = np.array(shape.get("points", []), dtype=np.int32)
        if pts.size == 0:
            continue
        if fill:
            cv2.fillPoly(overlay, [pts], color)
        else:
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
    return overlay

# === 5) Main ===
def main():
    imgs = get_all_image_paths(INPUT_DIR)
    print(f"[INFO] Found {len(imgs)} image(s) under {INPUT_DIR!r}")
    if not imgs:
        print("[WARN] No images to process. Check that INPUT_DIR is correct and contains files.")
        return

    for img_path in imgs:
        rel = os.path.relpath(img_path, INPUT_DIR)
        base, _ = os.path.splitext(os.path.basename(img_path))
        json_path = os.path.join(os.path.dirname(img_path), base + ".json")

        if not os.path.exists(json_path):
            print(f"[WARN] No JSON for {rel!r}; skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to load {rel!r}; skipping.")
            continue

        data = load_json(json_path)
        shapes = data.get("shapes", [])
        if not shapes:
            print(f"[WARN] JSON {base}.json has no 'shapes'; skipping.")
            continue

        annotated = draw_polygons_on_image(img, shapes, color=(0,255,0), thickness=2, fill=False)

        # replicate folder structure under OUTPUT_DIR
        out_subdir = os.path.join(OUTPUT_DIR, os.path.dirname(rel))
        os.makedirs(out_subdir, exist_ok=True)

        out_path = os.path.join(out_subdir, base + "_annotated.png")
        cv2.imwrite(out_path, annotated)
        print(f"[OK] {rel} → {os.path.relpath(out_path, SCRIPT_DIR)}")

if __name__ == "__main__":
    main()
