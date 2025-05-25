import os
import glob
import json
import cv2
import numpy as np

# === 1) EXPLICIT PATHS ===
# Make sure you run this script from the project root:
# C:\Users\alyan\PSE\PSM-Protech-Feasibility-Study

# Path to your images + JSONs:
INPUT_DIR = os.path.abspath(
    r"PSM-Protech-Feasibility-Study\Src\Model\falcon_r1\Dataset\prean"
)

# Where you want your per-polygon crops saved:
OUTPUT_DIR = os.path.abspath(
    r"PSM-Protech-Feasibility-Study\Src\Model\falcon_r1\bigcrop_output"
)

print(f"[INFO] INPUT_DIR:  {INPUT_DIR}")
print(f"[INFO] OUTPUT_DIR: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === 2) Find all images recursively ===
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
def find_images(root):
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return imgs


# === 3) JSON loader ===
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# === 4) Crop one polygon ===
def crop_polygon(img, pts, apply_mask=True):
    x, y, w, h = cv2.boundingRect(pts)
    crop = img[y : y + h, x : x + w].copy()

    if apply_mask:
        shifted = pts - np.array([[x, y]])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], 255)
        b, g, r = cv2.split(crop)
        rgba = cv2.merge([b, g, r, mask])
        return rgba
    return crop


# === 5) Main ===
def main():
    images = find_images(INPUT_DIR)
    print(f"[INFO] Found {len(images)} image(s) under '{INPUT_DIR}'")

    for img_path in images:
        rel = os.path.relpath(img_path, INPUT_DIR)
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(os.path.dirname(img_path), base + ".json")

        if not os.path.exists(json_path):
            print(f"[WARN] No JSON for '{rel}', skipping.")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERROR] Failed to load '{rel}', skipping.")
            continue

        data = load_json(json_path)
        shapes = data.get("shapes", [])
        if not shapes:
            print(f"[WARN] '{base}.json' has no shapes, skipping.")
            continue

        # Make per-image subfolder in OUTPUT_DIR
        sub_out = os.path.join(OUTPUT_DIR, os.path.dirname(rel))
        os.makedirs(sub_out, exist_ok=True)

        for i, shape in enumerate(shapes):
            pts = np.array(shape["points"], dtype=np.int32)
            if pts.size == 0:
                continue

            label = shape.get("label", "shape")
            # sanitize label
            label = "".join(c for c in label if c.isalnum() or c in ("-", "_")).strip()

            crop = crop_polygon(img, pts, apply_mask=True)
            out_name = f"{base}_{label}_{i:02d}.png"
            out_path = os.path.join(sub_out, out_name)

            cv2.imwrite(out_path, crop)
            print(f"[OK] {rel} → {os.path.relpath(out_path)}")

if __name__ == "__main__":
    main()
