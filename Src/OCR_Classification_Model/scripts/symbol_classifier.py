import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 1. Configuration ===
# === 1. Configuration ===
base_dir = r"C:\Users\Admin\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\Dataset"
input_folders = [
    os.path.join(base_dir, "PreAnnotated"),
    os.path.join(base_dir, "post_annotator")
]

img_size = (64, 64)  # Resize images to 64x64

# === 2. Load Images and Labels ===
images = []
labels = []

for folder in input_folders:
    if not os.path.exists(folder):
        continue
    print(f"Scanning folder: {folder}")  # To verify folders
    for filename in os.listdir(folder):
        print(f"Found file: {filename}")  # To verify file reading
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = filename.split('_')[0]  # Extract label from filename
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Couldn't load image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(img, img_size)
            img_flatten = img.flatten()  # Flatten to 1D vector
            images.append(img_flatten)
            labels.append(label)

print(f"\n✅ Total images loaded: {len(images)}")

# Safety check: if no images found, exit
if len(images) == 0:
    raise ValueError("No images found! Please check the 'PreAnnotated' and 'Annotator' folders.")

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Train a simple Random Forest Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === 4. Evaluate ===
y_pred = clf.predict(X_test)

print("\n🎯 Model Evaluation:")
print("---------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
