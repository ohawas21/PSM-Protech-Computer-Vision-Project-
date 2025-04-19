import cv2
import numpy as np
import os
import random

# Function to generate a random tolerance
def generate_random_tolerance():
    return f"{random.randint(1, 10)},{random.randint(1, 10)}"

# Load the target image where we'll search for symbols
target_image_path = "target_image.jpg"  # Replace with your image file
base_image = cv2.imread(target_image_path)
base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

# Path to the directory containing the symbol templates (e.g., straightness, flatness, etc.)
symbol_directory = "Datasets/OCR_Classification_Data/"  # Replace with the directory path containing the symbol images

# Threshold for template matching
threshold = 0.8  # Adjust based on how strict the matching should be

# Loop through all templates in the directory
for filename in os.listdir(symbol_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure it's an image file
        symbol_path = os.path.join(symbol_directory, filename)
        symbol_template = cv2.imread(symbol_path, 0)  # Load symbol in grayscale

        # Template matching to find regions in the target image matching the template
        result = cv2.matchTemplate(base_image_gray, symbol_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        # Label symbol based on the file name
        symbol_label = os.path.splitext(filename)[0]  # Use file name (without extension) as the label

        # Mark detections for the symbol on the image
        for pt in zip(*locations[::-1]):  # Switch x and y coordinates
            random_tolerance = generate_random_tolerance()
            cv2.rectangle(base_image, pt, (pt[0] + symbol_template.shape[1], pt[1] + symbol_template.shape[0]), (0, 255, 0), 2)
            cv2.putText(base_image, f"{symbol_label}: {random_tolerance}", (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display the annotated output
output_image_path = "output_annotated_image.jpg"
cv2.imwrite(output_image_path, base_image)
cv2.imshow("Annotated Image", base_image)
cv2.waitKey(0)
cv2.destroyAllWindows()