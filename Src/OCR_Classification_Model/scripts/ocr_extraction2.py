import re
import logging
from pathlib import Path
from io import BytesIO
import warnings

import cv2
import numpy as np
import camelot
import pandas as pd
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

# ─── Configuration ──────────────────────────────────────────────────────────────
IMAGE_DIR   = Path(r"PSM-Protech-Feasibility-Study\Src\OCR_Extraction_Model\Dataset")
PDF_DIR     = IMAGE_DIR.parent / "pdfs"
CSV_DIR     = IMAGE_DIR.parent / "csvs"
OUTPUT_CSV  = CSV_DIR / "extracted_numbers.csv"
# ────────────────────────────────────────────────────────────────────────────────



