#!/usr/bin/env python3
import sys
from pathlib import Path

# ── 1. Make sure project root is on sys.path ────────────────────────────────────
project_root = Path(__file__).parents[3]  
#  └— if this file is: …/Src/OCR_Extraction_Model/Scripts/ocr_pipeline_verification.py  
#     then parents[3] points at your repo root
sys.path.insert(0, str(project_root))

# ── 2. Import and run the pipeline ─────────────────────────────────────────────
from Src.OCR_Extraction_Model.Scripts.ocr_pipeline import OCRPipeline

pipeline = OCRPipeline(image_dir="Src/OCR_Extraction_Model/Dataset")
pipeline.convert_images_to_pdf()

# ── 3. Count check ─────────────────────────────────────────────────────────────
img_dir = Path("Src/OCR_Extraction_Model/Dataset")
img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}]

print(f"Images found:   {len(img_files)}")
print(f"PDF buffers:    {len(pipeline._pdf_buffers)}")
