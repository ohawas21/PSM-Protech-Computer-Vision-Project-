import json
import cv2
import numpy as np
from pathlib import Path

def find_images(root: Path, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")):
    """
    Recursively find all images under `root` matching the given extensions.
    Returns a list of Path objects.
    """
    imgs = []
    for ext in exts:
        imgs.extend(root.rglob(ext))
    return imgs


def load_json(path: Path):
    """
    Load JSON data from a file path and return the parsed object.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def crop_polygon(img: np.ndarray, pts: np.ndarray, apply_mask: bool = True) -> np.ndarray:
    """
    Crop a polygon defined by `pts` out of `img`.
    If `apply_mask` is True, returns an RGBA image with the polygon masked.
    """
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


def main():
    # Determine script and project structure
    SCRIPTS_DIR = Path(__file__).resolve().parent
    ROOT_DIR    = SCRIPTS_DIR.parent  # .../falcon_r1

    # Paths for input images and output crops
    INPUT_DIR  = ROOT_DIR / "Dataset" / "prean"
    OUTPUT_DIR = ROOT_DIR / "bigcrop_output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] INPUT_DIR:  {INPUT_DIR}")
    print(f"[INFO] OUTPUT_DIR: {OUTPUT_DIR}")

    # Find images
    images = find_images(INPUT_DIR)
    print(f"[INFO] Found {len(images)} image(s) under '{INPUT_DIR}'")

    for img_path in images:
        rel       = img_path.relative_to(INPUT_DIR)
        base      = img_path.stem
        json_path = img_path.with_suffix(".json")

        if not json_path.exists():
            print(f"[WARN] No JSON for '{rel}', skipping.")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERROR] Failed to load '{rel}', skipping.")
            continue

        data   = load_json(json_path)
        shapes = data.get("shapes", [])
        if not shapes:
            print(f"[WARN] '{base}.json' has no shapes, skipping.")
            continue

        # Create subfolder in output matching input structure
        sub_out = OUTPUT_DIR / rel.parent
        sub_out.mkdir(parents=True, exist_ok=True)

        for i, shape in enumerate(shapes):
            pts = np.array(shape.get("points", []), dtype=np.int32)
            if pts.size == 0:
                continue

            # Clean up label text
            label = shape.get("label", "shape")
            label = "".join(c for c in label if c.isalnum() or c in ("-", "_")).strip()

            # Crop and save
            crop     = crop_polygon(img, pts, apply_mask=True)
            out_name = f"{base}_{label}_{i:02d}.png"
            out_path = sub_out / out_name

            cv2.imwrite(str(out_path), crop)
            print(f"[OK] {rel} → {out_path.relative_to(ROOT_DIR)}")

if __name__ == "__main__":
    main()
