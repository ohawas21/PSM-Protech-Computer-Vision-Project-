import os
import pandas as pd
import logging
from ultralytics import YOLO

INPUT_FOLDER = 'falcon_r4'
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# Update the extract_data function to skip processing if the model is None
def extract_data():
    if model is None:
        logging.error("Skipping data extraction as the model could not be loaded.")
        return

    logging.info('Starting falcon_r4 data extraction...')
    excel_path = os.path.join(OUTPUT_FOLDER, 'final_output.xlsx')
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    for class_folder in os.listdir(INPUT_FOLDER):
        class_path = os.path.join(INPUT_FOLDER, class_folder)
        if os.path.isdir(class_path):
            logging.info(f'Processing folder: {class_folder}')
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                logging.info(f'Processing file: {file_name}')
                results = model.predict(source=file_path, save=False)  # Perform inference
                for i, result in enumerate(results):
                    # Save extracted data to Excel
                    output_excel_path = os.path.join(OUTPUT_FOLDER, f"{class_folder}_{i}.xlsx")
                    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                        result.pandas().xyxy.to_excel(writer, index=False)
            logging.info(f'Folder processed and data saved to {OUTPUT_FOLDER}: {class_folder}')
    writer.save()
    logging.info('falcon_r4 data extraction completed.')

if __name__ == "__main__":
    extract_data()
