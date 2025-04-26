import cv2
import os

"""
verify_annotations.py

Reads YOLO-format .txt annotation files alongside images,
draws the bounding boxes defined in the annotations,
and saves or displays the verification images.
"""

def verify_bounding_boxes(image_folder, output_folder):
    """
    For each image in image_folder that has a .txt annotation,
    draw the bounding box on the image and save it to output_folder.

    Args:
        image_folder (str): Path containing images and .txt files.
        output_folder (str): Path to save the verification images.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for fname in os.listdir(image_folder):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            continue

        img_path = os.path.join(image_folder, fname)
        txt_path = os.path.join(image_folder, f"{name}.txt")

        # Skip if no annotation file
        if not os.path.isfile(txt_path):
            print(f"Skipping {fname}: no annotation file found.")
            continue

        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Read annotation
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()

        # Draw each bounding box
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"Invalid annotation in {txt_path}: {line}")
                continue

            cls, cx, cy, bw, bh = parts
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))

            # Convert normalized to pixel coordinates
            box_w = int(bw * w)
            box_h = int(bh * h)
            box_x = int((cx * w) - box_w / 2)
            box_y = int((cy * h) - box_h / 2)

            # Draw rectangle
            cv2.rectangle(img,
                          (box_x, box_y),
                          (box_x + box_w, box_y + box_h),
                          (0, 255, 0), 2)
            cv2.putText(img, f"Class {cls}",
                        (box_x, box_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Save verification image
        out_path = os.path.join(output_folder, f"{name}_verified{ext}")
        cv2.imwrite(out_path, img)
        print(f"Saved verification image: {out_path}")


if __name__ == '__main__':
    # Example usage:
    IMAGE_DIR = r"C:\Users\Admin\psm\PSM-Protech-Feasibility-Study-1\Src\OCR_Classification_Model\Dataset\final_annotation"
    OUTPUT_DIR = IMAGE_DIR + "_verified"

    verify_bounding_boxes(IMAGE_DIR, OUTPUT_DIR)
    print("Verification complete.")