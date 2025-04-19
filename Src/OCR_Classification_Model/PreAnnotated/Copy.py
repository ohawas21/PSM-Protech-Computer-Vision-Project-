import os
import shutil

def duplicate_png_images(folder_path, num_copies=10):
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            base_name, ext = os.path.splitext(filename)

            for i in range(1, num_copies + 1):
                new_filename = f"{base_name}_{i}{ext}"
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_filename)
                shutil.copy2(src, dst)
                print(f"Created: {new_filename}")

# Folder path from your example
folder_path = "/Users/mugeshvaikundamani/Library/Mobile Documents/com~apple~CloudDocs/THRo/PSE/PSM-Protech-Feasibility-Study/Src/OCR_Classification_Model/Images"
duplicate_png_images(folder_path)