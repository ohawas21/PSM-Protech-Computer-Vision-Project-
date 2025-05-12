import os
import glob
import cv2

def annotate_yolo_images(input_folder, output_folder, num_files=10):
    """
    Reads YOLO-format .txt annotations and draws bounding boxes on images.
    Processes only the first `num_files` annotation files in `input_folder`
    and saves annotated images to `output_folder`.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get sorted list of annotation files
    annotation_paths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))

    # Process only the first `num_files` annotations
    for ann_path in annotation_paths[:num_files]:
        base = os.path.splitext(os.path.basename(ann_path))[0]

        # Find corresponding image file (common extensions)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(input_folder, base + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"[WARN] No image found for annotation '{ann_path}', skipping.")
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to load image '{img_path}', skipping.")
            continue
        height, width = img.shape[:2]

        # Read YOLO annotation
        with open(ann_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Malformed line
            class_id, x_center, y_center, w_rel, h_rel = parts
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            w_rel = float(w_rel)
            h_rel = float(h_rel)

            # Convert relative coords to pixels
            x1 = int((x_center - w_rel / 2) * width)
            y1 = int((y_center - h_rel / 2) * height)
            x2 = int((x_center + w_rel / 2) * width)
            y2 = int((y_center + h_rel / 2) * height)

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(class_id), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save annotated image
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        print(f"[INFO] Saved annotated image to '{output_path}'")


if __name__ == '__main__':
    # TODO: Update these paths to your actual folders
    input_folder = 'path/to/your/input/folder'
    output_folder = 'path/to/your/output/folder'
    # Only the first 10 annotation files will be processed
    annotate_yolo_images(input_folder, output_folder, num_files=10)
