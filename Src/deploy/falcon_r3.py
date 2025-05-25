import os
import logging
import cv2
from ultralytics import YOLO

# === Configuration ===
MODEL_PATH = "models/falcon_r3.pt"  # Must be trained using task='classify'
INPUT_FOLDER = "falcon_r3"          # Folder with unclassified images
OUTPUT_FOLDER = "falcon_r4"         # Output folder organized by predicted class

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='progress.log',
    filemode='a'
)

# Load the classification model
if not os.path.exists(MODEL_PATH):
    logging.error(f"❌ Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    logging.info(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def classify_images():
    logging.info("🚀 Starting YOLO classification...")

    for file_name in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            results = model.predict(source=file_path, save=False)
            pred_class_index = results[0].probs.top1
            pred_class_name = results[0].names[pred_class_index]

            # Save image to predicted class folder
            output_class_folder = os.path.join(OUTPUT_FOLDER, pred_class_name)
            os.makedirs(output_class_folder, exist_ok=True)
            output_path = os.path.join(output_class_folder, file_name)

            cv2.imwrite(output_path, cv2.imread(file_path))
            logging.info(f"✅ {file_name} → {pred_class_name}")
        except Exception as e:
            logging.error(f"❌ Error processing {file_name}: {e}")

    logging.info("🎉 Classification complete.")

if __name__ == "__main__":
    classify_images()
