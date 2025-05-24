# blueprint_crop.py

import cv2
import numpy as np
import os
import argparse

def extract_and_save_polygons(input_image_path, output_dir):
    """
    Load a blueprint image, detect purple annotation outlines, extract each polygon region,
    and save them as individual image files.
    """
    # Load image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {input_image_path}")

    # Convert to HSV for color thresholding of purple outlines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define purple color range (adjust if needed)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Find contours of the purple annotations
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0

    for i, cnt in enumerate(contours):
        # Approximate contour to a polygon to clean up
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Create mask for the current polygon
        poly_mask = np.zeros_like(mask)
        cv2.fillPoly(poly_mask, [poly], 255)

        # Apply mask to the original image
        extracted = cv2.bitwise_and(image, image, mask=poly_mask)

        # Crop to bounding rectangle of the polygon
        x, y, w, h = cv2.boundingRect(poly)
        if w < 10 or h < 10:
            continue  # skip very small regions
        crop = extracted[y:y+h, x:x+w]

        # Save cropped polygon image
        out_path = os.path.join(output_dir, f"polygon_{i:03d}.png")
        cv2.imwrite(out_path, crop)
        saved_count += 1

    print(f"[INFO] Saved {saved_count} polygon images to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract annotated polygon regions from a blueprint image"
    )
    parser.add_argument("input_image", help="Path to the blueprint image file")
    parser.add_argument(
        "--output_dir",
        default=r"PSM-Protech-Feasibility-Study\\Documentation\\Testing",
        help="Directory to save extracted polygon images"
    )
    args = parser.parse_args()

    extract_and_save_polygons(args.input_image, args.output_dir)

if __name__ == "__main__":
    main()
