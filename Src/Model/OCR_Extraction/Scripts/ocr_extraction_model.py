#!/usr/bin/env python3
# blueprint_crop.py

import cv2
import numpy as np
import os
import argparse

def extract_polygons_color(image, hsv_lower, hsv_upper):
    """Extract contours by HSV color mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def extract_polygons_edge(image, canny_thresh1=50, canny_thresh2=150):
    """Extract contours by Canny edge detection → dilation → findContours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)
    # dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def save_contour_crops(image, contours, output_dir, prefix):
    """Mask, crop & save each contour as its own image."""
    saved = 0
    mask_shape = image.shape[:2]
    for i, cnt in enumerate(contours):
        # approximate to reduce vertex count
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        # skip tiny
        x,y,w,h = cv2.boundingRect(poly)
        if w<10 or h<10: continue

        # render mask & apply
        mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cropped = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]

        out_path = os.path.join(output_dir, f"{prefix}_poly_{i:03d}.png")
        cv2.imwrite(out_path, cropped)
        saved += 1
    return saved

def process_image(path, out_dir):
    image = cv2.imread(path)
    if image is None:
        print(f"[WARN] Could not open {path}")
        return 0

    base = os.path.splitext(os.path.basename(path))[0]
    # first try color
    hsv_lo = np.array([120,50,50])
    hsv_hi = np.array([160,255,255])
    cnts = extract_polygons_color(image, hsv_lo, hsv_hi)
    if len(cnts)==0:
        # fallback to edge
        cnts = extract_polygons_edge(image)
        method = "edge"
    else:
        method = "color"

    n = save_contour_crops(image, cnts, out_dir, base)
    print(f"[INFO] {base}: {n} polygons ({method})")
    return n

def main():
    p = argparse.ArgumentParser(
        description="Batch-extract annotated polygons (color→edge fallback)."
    )
    p.add_argument(
        "--input_dir",
        default=r"PSM-Protech-Feasibility-Study\Src\Model\falcon_r1\dataset\falconv1_annote\train\images",
        help="Folder of input images"
    )
    p.add_argument(
        "--output_dir",
        default=r"PSM-Protech-Feasibility-Study\Documentation\Testing",
        help="Where to save crops"
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    for fn in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(fn.lower())[1] not in exts:
            continue
        total += process_image(
            os.path.join(args.input_dir, fn),
            args.output_dir
        )
    print(f"[DONE] Total saved: {total}")

if __name__=="__main__":
    main()
