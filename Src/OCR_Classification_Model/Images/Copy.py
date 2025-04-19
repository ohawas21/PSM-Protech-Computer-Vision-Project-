import os
import shutil
import random

def duplicate_images(folder_path, min_copies=10, max_copies=15):
    # Make sure folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("form_") and filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
            base_name, ext = os.path.splitext(filename)
            num_copies = random.randint(min_copies, max_copies)

            for i in range(1, num_copies + 1):
                new_filename = f"{base_name}_{i}{ext}"
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_filename)
                shutil.copy2(src, dst)
                print(f"Created: {new_filename}")

# Example usage:
folder_path = "/Users/mugeshvaikundamani/Library/Mobile Documents/com~apple~CloudDocs/THRo/PSE/PSM-Protech-Feasibility-Study/Src/OCR_Classification_Model/Images"  # Replace with the actual path
duplicate_images(folder_path)