import os
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# === CONFIGURATION ===
MODEL_PATH = r"C:\Users\Admin\psm\PSM-Protech-Feasibility-Study\Src\Model\OCR_Classification\Model\best.pt"  # use raw string
TEST_DIR = r"C:\Users\Admin\psm\PSM-Protech-Feasibility-Study\Src\Model\OCR_Classification\Dataset\Custom_Testing"
RESULT_FILE = "classification_results.txt"
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
DEVICE = "cpu"  # use 'cpu' if 'mps' is not supported on Windows

# Manual mapping from class name to index (matching your trained model)
CLASS_NAMES = {
    "angularity": 0,
    "circular": 1,
    "concentricity": 2,
    "line": 3,
    "parallelism": 4,
    "perpendicularity": 5,
    "surface": 6,
    "total": 7
}

# For reverse lookup: index to class name
INDEX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

def infer_class_from_filename(filename):
    return filename.split("_")[0].lower()

def load_images(directory):
    image_paths = []
    ground_truths = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXTS):
            image_path = os.path.join(directory, filename)
            true_label = infer_class_from_filename(filename)
            image_paths.append(image_path)
            ground_truths.append(true_label)
    return image_paths, ground_truths

def main():
    print("🚀 Loading model...")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    print("🔍 Loading test images...")
    image_paths, ground_truths = load_images(TEST_DIR)
    print(f"✅ Found {len(image_paths)} test images.")

    correct = 0
    total = len(image_paths)
    predictions = []

    print("🔎 Running inference...")
    for path, true_label in tqdm(zip(image_paths, ground_truths), total=total):
        results = model(path)
        pred_index = results[0].probs.top1
        pred_label = INDEX_TO_CLASS[pred_index]
        is_correct = pred_label == true_label
        predictions.append((os.path.basename(path), true_label, pred_label, is_correct))
        correct += int(is_correct)

    accuracy = correct / total * 100

    with open(RESULT_FILE, "w") as f:
        for img, gt, pred, ok in predictions:
            f.write(f"{img}\tGT: {gt}\tPred: {pred}\t{'✅' if ok else '❌'}\n")
        f.write(f"\nOverall Accuracy: {accuracy:.2f}%\n")

    print(f"\n📄 Results written to: {RESULT_FILE}")
    print(f"✅ Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
