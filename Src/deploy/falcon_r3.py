import os
import logging
from ultralytics import YOLO
import cv2  # Ensure cv2 is imported for image processing

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

INPUT_FOLDER = 'falcon_r3'
OUTPUT_FOLDER = 'falcon_r4'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Update the classify_images function to skip processing if the model is None
def classify_images():
    if model is None:
        logging.error("Skipping classification as the model could not be loaded.")
        return

    logging.info('Starting falcon_r3 classification...')
    for class_folder in os.listdir(INPUT_FOLDER):
        class_path = os.path.join(INPUT_FOLDER, class_folder)
        if os.path.isdir(class_path):
            logging.info(f'Processing folder: {class_folder}')
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                logging.info(f'Processing file: {file_name}')
                results = model.predict(source=file_path, save=False)  # Perform inference
                for i, result in enumerate(results):
                    # Save classified images
                    output_class_folder = os.path.join(OUTPUT_FOLDER, str(result.names[0]))
                    os.makedirs(output_class_folder, exist_ok=True)
                    output_path = os.path.join(output_class_folder, file_name)
                    cv2.imwrite(output_path, cv2.imread(file_path))
            logging.info(f'Folder processed and saved to {OUTPUT_FOLDER}: {class_folder}')
    logging.info('falcon_r3 classification completed.')

if __name__ == "__main__":
    classify_images()
