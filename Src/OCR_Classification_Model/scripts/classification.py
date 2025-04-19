import cv2
import os
import random
import numpy as np

# Function to generate a random float as a string (e.g., '6.7')
def generate_random_tolerance():
    return f"{round(random.uniform(5.0, 10.0), 1)}"

# Base paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

images_folder = os.path.join(base_path, "PreAnnotated")
output_folder = os.path.join(base_path, "post_annotated")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Check if images_folder exists
if not os.path.exists(images_folder):
    raise FileNotFoundError(f"Directory not found: {images_folder}")

# Load image files
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".png"))]
if not image_files:
    raise FileNotFoundError(f"No images found in: {images_folder}")

# Process each image
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    print(f"Processing: {image_file}")

    # Simulate 1–3 "symbols" randomly for demo
    for _ in range(random.randint(1, 3)):
        # Generate random position and box size
        w, h = random.randint(30, 80), random.randint(30, 60)

        # Ensure valid range for x and y
        x_max = max(0, width - w - 100)
        y_max = max(0, height - h - 20)

        # If x_max or y_max is non-positive, set x and y to 0 to avoid invalid ranges
        x = random.randint(0, x_max) if x_max > 0 else 0
        y = random.randint(0, y_max) if y_max > 0 else 0

        label = "a"
        tolerance = generate_random_tolerance()

        # Draw black rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Put the label above the rectangle
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Put the tolerance value to the right of the rectangle
        cv2.putText(image, tolerance, (x + w + 10, y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Optional vertical separator line
        cv2.line(image, (x + w + 5, y), (x + w + 5, y + h), (0, 0, 0), 1)

    # Save the result
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)
    print(f"Saved to: {output_path}")

print("All images processed and saved.")
