import cv2
import os
import random
import numpy as np

# Function to generate a two-digit decimal number as a string (e.g., '53.4')
def generate_random_decimal():
    return f"{round(random.uniform(10.0, 99.9), 1)}"

# Base directory for OCR Classification Model
base_dir = "/Users/mugeshvaikundamani/Library/Mobile Documents/com~apple~CloudDocs/THRo/PSE/PSM-Protech-Feasibility-Study/Src/OCR_Classification_Model"

# Identify input folders
input_folders = [
    os.path.join(base_dir, "PreAnnotated"),
    os.path.join(base_dir, "Annotator")
]
# Retain only those that exist
input_folders = [folder for folder in input_folders if os.path.exists(folder)]
if not input_folders:
    raise FileNotFoundError("No valid input folders found under OCR_Classification_Model.")

# Ensure output folder exists
output_folder = os.path.join(base_dir, "post_annotator")
os.makedirs(output_folder, exist_ok=True)

# Process images from each input folder
for folder in input_folders:
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: couldn't load {img_path}")
            continue

        h, w = img.shape[:2]
        seg_w = w // 3  # width of each of three equal segments

        # Draw and annotate segments
        for i in range(3):
            x0 = i * seg_w
            x1 = w if i == 2 else (x0 + seg_w)
            # Draw rectangle border for each segment
            cv2.rectangle(img, (x0, 0), (x1, h), (0, 0, 0), 2)

            # Only annotate in 2nd and 3rd segments
            if i in (1, 2):
                decimal_text = generate_random_decimal()
                text_size, _ = cv2.getTextSize(decimal_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_x = x0 + (seg_w - text_size[0]) // 2
                text_y = h // 2 + text_size[1] // 2
                cv2.putText(img, decimal_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Save annotated image
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, img)
        print(f"Processed and saved: {out_path}")

print("Done processing all images.")
