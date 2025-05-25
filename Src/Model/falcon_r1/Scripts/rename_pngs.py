import os
from pathlib import Path

# Determine directory relative to this script's location
script_dir = Path(__file__).resolve().parent
# bigcrop_output folder is one level up from the Scripts folder
directory = script_dir.parent / "bigcrop_output"

print(f"Looking for PNG files in: {directory}")

# Ensure the directory exists
if not directory.exists() or not directory.is_dir():
    raise FileNotFoundError(f"Directory not found: {directory}")

# Get all PNG files sorted alphabetically
png_files = sorted(directory.glob("*.png"))
print(f"Found {len(png_files)} PNG files to rename.")

# If no files found, exit
if not png_files:
    print("No PNG files to rename. Exiting.")
    exit(0)

# Rename each file to a simple numeric sequence: 1.png, 2.png, ...
for index, file_path in enumerate(png_files, start=1):
    new_name = f"{index}.png"
    new_path = directory / new_name

    # Skip if the target name already exists
    if new_path.exists():
        print(f"Skipping {file_path.name}: target name {new_name} already exists.")
        continue

    file_path.rename(new_path)
    print(f"Renamed {file_path.name} -> {new_name}")

print("Done renaming PNG files.")
