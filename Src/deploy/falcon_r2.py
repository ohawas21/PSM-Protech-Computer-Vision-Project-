import os
import torch
import cv2
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='progress.log', filemode='a')

# Dynamically load the model based on the script name
script_name = os.path.splitext(os.path.basename(__file__))[0]
MODEL_PATH = f'models/{script_name}.pt'

INPUT_FOLDER = 'falcon_r2_preprocess'
OUTPUT_FOLDER = 'falcon_r3'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to check the availability of all required models
def check_all_models():
    """Check if all required models are available."""
    required_models = [
        'models/falcon_r1.pt',
        'models/falcon_r2.pt',
        'models/falcon_r3.pt'  # Exclude falcon_r4.pt as it is not a model
    ]

    missing_models = [model for model in required_models if not os.path.exists(model)]

    if missing_models:
        logging.error(f"Missing models: {', '.join(missing_models)}")
        # Clean up all intermediate folders
        folders_to_clean = ['falcon_r1_preprocess', 'falcon_r2_preprocess', 'falcon_r3', 'falcon_r4']
        for folder in folders_to_clean:
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # Remove file
                except Exception as e:
                    logging.error(f"Error cleaning up {file_path}: {e}")
        return False  # Indicate that models are missing
    return True  # All models are available

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    logging.error(f'Model file not found: {MODEL_PATH}')
    for file_name in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Remove file
        except Exception as e:
            logging.error(f'Error cleaning up {file_path}: {e}')
    # Log the error instead of raising an exception
    logging.error(f'Model file not found: {MODEL_PATH}')

# Load the YOLO model using the ultralytics library
try:
    model = YOLO(MODEL_PATH)
except FileNotFoundError as e:
    logging.error(f"Failed to load YOLO model: {e}")
    model = None  # Set model to None to indicate failure

def process_images():
    if model is None:
        logging.error("Skipping image processing as the model could not be loaded.")
        return

    logging.info('Starting falcon_r2 processing...')
    for file_name in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        logging.info(f'Processing file: {file_name}')
        results = model.predict(source=file_path, save=False)  # Perform inference
        for i, result in enumerate(results):
            # Extract bounding boxes and save cropped images
            for box in result.boxes.xyxy:
                xmin, ymin, xmax, ymax = map(int, box[:4])
                cropped_img = cv2.imread(file_path)[ymin:ymax, xmin:xmax]
                output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file_name)[0]}_{i}.png")
                cv2.imwrite(output_path, cropped_img)
        logging.info(f'File processed and saved to {OUTPUT_FOLDER}: {file_name}')
    logging.info('falcon_r2 processing completed.')

if __name__ == "__main__":
    process_images()
