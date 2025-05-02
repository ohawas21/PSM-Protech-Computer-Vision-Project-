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
from typing import Optional


# ─── Configuration ──────────────────────────────────────────────────────────────
IMAGE_DIR   = Path(r"PSM-Protech-Feasibility-Study\Src\OCR_Extraction_Model\Dataset")
PDF_DIR     = IMAGE_DIR.parent / "pdfs"
CSV_DIR     = IMAGE_DIR.parent / "csvs"
OUTPUT_CSV  = CSV_DIR / "extracted_numbers.csv"
# ────────────────────────────────────────────────────────────────────────────────

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────────────────────

# Suppress Camelot warnings for image-based pages
warnings.filterwarnings('ignore', category=UserWarning)

# Tesseract config: only digits and dot
TESS_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789."

class TableOCRExtractor:
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.pdf_dir   = PDF_DIR
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self.results = []  # list of dicts: { 'source_image', 'col2', 'col3' }

    def convert_image_to_pdf(self, img_path: Path) -> Path:
        buf = BytesIO()
        Image.open(img_path).convert("RGB").save(buf, format="PDF")
        buf.seek(0)
        pdf_path = self.pdf_dir / f"{img_path.stem}.pdf"
        pdf_path.write_bytes(buf.getvalue())
        return pdf_path

    def try_camelot(self, pdf_path: Path, name: str):
        try:
            tables = camelot.read_pdf(str(pdf_path), flavor="stream", pages="1")
            if tables:
                df = tables[0].df
                c2 = df.iloc[0,1].strip()
                c3 = df.iloc[0,2].strip()
                logger.info("  ✔ Camelot parsed %s → %s, %s", name, c2, c3)
                return c2, c3
        except Exception as e:
            logger.warning("  Camelot error on %s: %s", name, e)
        return None

    def render_pdf_to_image(self, pdf_path: Path) -> Image.Image | None:
        try:
            pages = convert_from_bytes(pdf_path.read_bytes(), dpi=300)
            return pages[0] if pages else None
        except Exception as e:
            logger.error("  PDF rendering error for %s: %s", pdf_path.name, e)
            return None
        
    def crop_columns(self, img: Image.Image) -> tuple[Image.Image, Image.Image]:
        W, H = img.size
        x1 = int(W * 0.20)
        x2 = int(W * 0.60)
        x3 = int(W * 0.98)
        col2_img = img.crop((x1, 0, x2, H))
        col3_img = img.crop((x2, 0, x3, H))
        return col2_img, col3_img
        
    def preprocess(self, region: Image.Image) -> Image.Image:
        arr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(arr, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(th)
        up = cv2.resize(inv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(up)
    
    def ocr_region(self, region: Image.Image, name: str, label: str) -> Optional[str]:
        prep = self.preprocess(region)
        txt = pytesseract.image_to_string(prep, config=TESS_CONFIG).strip()
        snippet = txt.replace("\n", " ")[:40]
        logger.info("    OCR[%s] snippet: '%s...';", label, snippet)
        m = re.search(r"\d+\.\d+", txt)
        if m:
            val = m.group(0)
            logger.info("    → %s = %s", label, val)
            return val
        logger.warning("    ⚠️ %s region %s: no decimal found", name, label)
        return None


     




