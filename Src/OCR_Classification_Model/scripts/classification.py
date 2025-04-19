import cv2
import os
import random
import numpy as np

# Function to generate a random tolerance
def generate_random_tolerance():
    """Generates a random tolerance value."""
    return f"{random.randint(1, 10)},{random.randint(1, 10)}"

# Paths to folders
images_folder = "PSM-Protech-Feasibility-Study-1/Src/OCR_Classification_Model/PreAnnotated"  # Folder containing pre-annotated images
template_directory = "PSM-Protech-Feasibility-Study-1/Src/OCR_Classification_Model/Images"  # Folder containing geometric symbol templates
output_folder = "Src/OCR_Classification_Model/post_annotated"  # Folder for saving annotated output images
os.makedirs(output_folder, exist_ok=True)  # Create the output directory if it doesn't exist

# Template matching threshold
threshold = 0.8  # Adjust based on sensitivity, higher is stricter

# Load all template images from the specified directory
templates = {}
for filename in os.listdir(template_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only image files
        template_name = os.path.splitext(filename)[0]  # Use filename (without extension) as label
        templates[template_name] = cv2.imread(os.path.join(template_directory, filename), 0)  # Load as grayscale

# Select 5 random images from the `PreAnnotated` folder
image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg") or f.endswith(".png")]
if len(image_files) < 5:
    print(f"Not enough images in the folder. Found only {len(image_files)} image(s).")
    selected_images = image_files
else:
    selected_images = random.sample(image_files, 5)  # Randomly select 5 images

# Process each selected image
for image_file in selected_images:
    image_path = os.path.join(images_folder, image_file)
    base_image = cv2.imread(image_path)  # Load image
    base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for template matching

    print(f"Processing image: {image_file}...")
    detected_symbols = []

    # Iterate through all templates and perform template matching
    for label, template in templates.items():
        result = cv2.matchTemplate(base_image_gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)  # Get coordinates where matches meet the threshold

        # Annotate detections on the image
        for pt in zip(*locations[::-1]):  # Reverse x and y coordinates
            random_tolerance = generate_random_tolerance()  # Generate random tolerance
            x, y = pt
            h, w = template.shape[:2]
            cv2.rectangle(base_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(
                base_image,
                f"{label}: {random_tolerance}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            detected_symbols.append((label, (x, y, w, h), random_tolerance))

    # Save the annotated image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, base_image)
    print(f"Annotated output saved at: {output_path}")

print("Processing completed for 5 random images!")