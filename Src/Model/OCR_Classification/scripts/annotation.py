import cv2
import os
import numpy as np

def create_yolo_annotation(x, y, w, h, img_width, img_height):
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

def extract_and_annotate_symbol(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    height, width = img.shape[:2]
    box_width = width // 3

    # Focus only on the first third
    symbol_region = img[0:height, 0:box_width]

    gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    possible_symbols = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Filter tiny noise
            possible_symbols.append((x, y, w, h))

    if not possible_symbols:
        raise ValueError("No symbol found!")

    possible_symbols = sorted(possible_symbols, key=lambda b: b[0])

    x, y, w, h = possible_symbols[0]

    # Coordinates relative to the full image
    full_x = x
    full_y = y
    full_w = w
    full_h = h

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save YOLO annotation file
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    yolo_annotation = create_yolo_annotation(full_x, full_y, full_w, full_h, width, height)

    annotation_path = os.path.join(os.path.dirname(image_path), f"{name}.txt")
    with open(annotation_path, "w") as f:
        f.write(yolo_annotation)

    print(f"Saved YOLO annotation for {base_filename} at {annotation_path}")

    # Optional: Save a verification image (draw bounding box)
    verify_img = img.copy()
    cv2.rectangle(verify_img, (full_x, full_y), (full_x+full_w, full_y+full_h), (0,255,0), 2)
    verify_path = os.path.join(output_dir, f"{name}_verify{ext}")
    cv2.imwrite(verify_path, verify_img)


def batch_extract_symbols(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            extract_and_annotate_symbol(file_path, output_folder)

if __name__ == "__main__":
    input_folder = r"C:\\Users\\Admin\\psm pro\\PSM-Protech-Feasibility-Study\\Src\\OCR_Classification_Model\\Dataset\\final_annotation"
    output_folder = "./output_symbols"

    batch_extract_symbols(input_folder, output_folder)
