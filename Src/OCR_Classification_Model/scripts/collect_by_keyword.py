<<<<<<< HEAD
import os
import shutil
from pathlib import Path

# Define absolute source and destination directories
SRC_DIR = Path(r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\post_annotator")
DST_DIR = Path(r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\final_annotation")

# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# List of keywords to categorize images
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

# Initialize counters for each keyword
counters = {kw: 1 for kw in KEYWORDS}


def find_keyword_in_name(name: str, keywords: list):
    """
    Return the first matching keyword found in the filename (case-insensitive), or None.
    """
    lower = name.lower()
    for kw in keywords:
        if kw in lower:
            return kw
    return None


def rename_and_copy_images(src_dir: Path, dst_dir: Path, keywords: list):
    """
    Scan all image files in src_dir, detect keyword in filename,
    rename to keyword_<n>.<ext>, and copy to dst_dir.
    """
    if not src_dir.is_dir():
        print(f"❌ Source directory not found: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if not files:
        print(f"ℹ️ No images found in source directory: {src_dir}")
        return

    for file in files:
        keyword = find_keyword_in_name(file.stem, keywords)
        if keyword:
            count = counters[keyword]
            new_name = f"{keyword}_{count}{file.suffix.lower()}"
            target = dst_dir / new_name
            shutil.copy2(file, target)
            print(f"✅ Copied: {file.name} → {new_name}")
            counters[keyword] += 1
        else:
            print(f"⚠️ No keyword match for file: {file.name}")


if __name__ == "__main__":
    print(f"🔍 Scanning source directory: {SRC_DIR}")
    print(f"🗝️ Using keywords: {KEYWORDS}")
    print(f"📂 Output directory: {DST_DIR}\n")
    rename_and_copy_images(SRC_DIR, DST_DIR, KEYWORDS)



=======
import os
import shutil
from pathlib import Path

# Define absolute source and destination directories
SRC_DIR = Path(r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\post_annotator")
DST_DIR = Path(r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\final_annotation")

# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# List of keywords to categorize images
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

# Initialize counters for each keyword
counters = {kw: 1 for kw in KEYWORDS}


def find_keyword_in_name(name: str, keywords: list):
    """
    Return the first matching keyword found in the filename (case-insensitive), or None.
    """
    lower = name.lower()
    for kw in keywords:
        if kw in lower:
            return kw
    return None


def rename_and_copy_images(src_dir: Path, dst_dir: Path, keywords: list):
    """
    Scan all image files in src_dir, detect keyword in filename,
    rename to keyword_<n>.<ext>, and copy to dst_dir.
    """
    if not src_dir.is_dir():
        print(f"❌ Source directory not found: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if not files:
        print(f"ℹ️ No images found in source directory: {src_dir}")
        return

    for file in files:
        keyword = find_keyword_in_name(file.stem, keywords)
        if keyword:
            count = counters[keyword]
            new_name = f"{keyword}_{count}{file.suffix.lower()}"
            target = dst_dir / new_name
            shutil.copy2(file, target)
            print(f"✅ Copied: {file.name} → {new_name}")
            counters[keyword] += 1
        else:
            print(f"⚠️ No keyword match for file: {file.name}")


if __name__ == "__main__":
    print(f"🔍 Scanning source directory: {SRC_DIR}")
    print(f"🗝️ Using keywords: {KEYWORDS}")
    print(f"📂 Output directory: {DST_DIR}\n")
    rename_and_copy_images(SRC_DIR, DST_DIR, KEYWORDS)



>>>>>>> dac556b2e3a093c229aca5c71cdbb7081c327a13
