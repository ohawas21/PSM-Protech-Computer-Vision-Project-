import os
from PIL import Image
import cv2
import torch
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='progress.log', filemode='a')

# Dynamically load the model based on the script name
script_name = os.path.splitext(os.path.basename(__file__))[0]
MODEL_PATH = f'models/{script_name}.pt'

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    logging.error(f'Model file not found: {MODEL_PATH}')

# Load the YOLO model using the ultralytics library
try:
    model = YOLO(MODEL_PATH)
except FileNotFoundError as e:
    logging.error(f"Failed to load YOLO model: {e}")
    model = None  # Set model to None to indicate failure

INPUT_FOLDER = 'falcon_r1_preprocess'
OUTPUT_FOLDER = 'falcon_r2_preprocess'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Update the preprocess_and_infer function to skip processing if the model is None
def preprocess_and_infer():
    if model is None:
        logging.error("Skipping preprocessing as the model could not be loaded.")
        return

    logging.info('Starting falcon_r1 preprocessing...')
    for file_name in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        logging.info(f'Processing file: {file_name}')
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            # Convert to uniform resolution
            if file_name.lower().endswith('.pdf'):
                # Convert PDF to images (first page only for simplicity)
                from pdf2image import convert_from_path
                images = convert_from_path(file_path, dpi=300)
                for i, image in enumerate(images):
                    image.save(os.path.join(INPUT_FOLDER, f"{file_name}_{i}.png"))
                continue
            else:
                img = Image.open(file_path)
                img = img.resize((640, 640))
                img.save(file_path)

            # YOLO inference
            results = model.predict(source=file_path, save=False)  # Perform inference
            for i, result in enumerate(results):
                # Extract bounding boxes and save cropped images
                for box in result.boxes.xyxy:
                    xmin, ymin, xmax, ymax = map(int, box[:4])
                    cropped_img = cv2.imread(file_path)[ymin:ymax, xmin:xmax]
                    output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file_name)[0]}_{i}.png")
                    cv2.imwrite(output_path, cropped_img)
            logging.info(f'File processed and saved to {OUTPUT_FOLDER}: {file_name}')
    logging.info('falcon_r1 preprocessing completed.')

if __name__ == "__main__":
    preprocess_and_infer()
