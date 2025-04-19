import os
import shutil
from pathlib import Path

# Determine base directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # assumes script is in .../scripts/

# Attempt to locate the source directory (handles multiple naming conventions)
POSSIBLE_SRC_NAMES = ["post_annotator", "postannotated"]
SRC_DIR = None
for name in POSSIBLE_SRC_NAMES:
    candidate = BASE_DIR / name
    if candidate.is_dir():
        SRC_DIR = candidate
        break
if SRC_DIR is None:
    # fallback: pick any directory starting with "post"
    for sub in BASE_DIR.iterdir():
        if sub.is_dir() and sub.name.lower().startswith("post"):
            SRC_DIR = sub
            break
if SRC_DIR is None:
    print(f"❌ Could not find a source directory under {BASE_DIR}.")
    exit(1)

# Define destination directory
DST_DIR = BASE_DIR / "final_annotation"

# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# Explicit list of keyword folders to process
KEYWORDS = [
    "surface",
    "coxciality",
    "line",
    "concentricity",
    "symmetrity",
    "angularity",
    "parallelism",
    "perpendicularity",
    "circular",
    "total"
]


def rename_and_copy_images(src_dir: Path, dst_dir: Path, keywords: list):
    """
    For each keyword folder under src_dir, rename all images to keyword_<n>.<ext>
    and copy them into dst_dir.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    for keyword in keywords:
        folder = src_dir / keyword
        if not folder.is_dir():
            print(f"⚠️ Skipping '{keyword}': folder not found at {folder}")
            continue

        print(f"📁 Processing '{keyword}' in {folder}")
        count = 1
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in IMAGE_EXTS and file.is_file():
                new_name = f"{keyword}_{count}{file.suffix.lower()}"
                target = dst_dir / new_name
                shutil.copy2(file, target)
                print(f"Copied: {file.name} → {new_name}")
                count += 1

        if count == 1:
            print(f"ℹ️ No images found for '{keyword}' in {folder}")


if __name__ == "__main__":
    print(f"🔍 Scanning source directory: {SRC_DIR}")
    print(f"🗝️ Processing keywords: {KEYWORDS}\n")
    print(f"📂 Saving renamed images to: {DST_DIR}\n")

    rename_and_copy_images(SRC_DIR, DST_DIR, KEYWORDS)
