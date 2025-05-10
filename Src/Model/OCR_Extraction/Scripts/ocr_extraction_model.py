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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

TESS_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789."

class TableOCRExtractor:
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.pdf_dir   = PDF_DIR; self.pdf_dir.mkdir(parents=True, exist_ok=True)
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self.results = []  # each a dict with keys: source_image, col1, col2, …

    def convert_image_to_pdf(self, img_path: Path) -> Path:
        buf = BytesIO()
        Image.open(img_path).convert("RGB").save(buf, format="PDF")
        pdf_path = self.pdf_dir / f"{img_path.stem}.pdf"
        pdf_path.write_bytes(buf.getvalue())
        return pdf_path

    def try_camelot(self, pdf_path: Path, name: str) -> list[str] | None:
        try:
            tables = camelot.read_pdf(str(pdf_path), flavor="stream", pages="1")
            if tables:
                df = tables[0].df
                # skip column 0 (symbols), grab every other column in row 0
                values = []
                for col in range(1, df.shape[1]):
                    txt = df.iloc[0, col].strip()
                    if re.match(r"\d+\.\d+", txt):
                        values.append(txt)
                if values:
                    logger.info("  ✔ Camelot parsed %s → %s", name, values)
                    return values
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

    def fallback_ocr(self, img: Image.Image, name: str) -> list[str]:
        # crop off the first ~20% where the symbols live:
        W, H = img.size
        data_region = img.crop((int(W * 0.20), 0, W, H))
        # preprocess (reuse your existing routine):
        arr = cv2.cvtColor(np.array(data_region), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(arr, (3,3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(th)
        up = cv2.resize(inv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        txt = pytesseract.image_to_string(up, config=TESS_CONFIG)
        found = re.findall(r"\d+\.\d+", txt)
        logger.info("    Fallback OCR on %s → %s", name, found)
        return found

    def run(self):
        imgs = sorted(self.image_dir.glob("*.*"))
        imgs = [p for p in imgs if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}]
        logger.info(f"Processing {len(imgs)} images…")

        for img_path in imgs:
            name = img_path.stem
            logger.info(f"→ {name}")

            pdf = self.convert_image_to_pdf(img_path)
            values = self.try_camelot(pdf, name)

            if values is None:
                page_img = self.render_pdf_to_image(pdf)
                if page_img:
                    values = self.fallback_ocr(page_img, name)

            if values:
                # build a record with dynamic columns
                rec = {"source_image": name}
                for i, v in enumerate(values, start=1):
                    rec[f"col{i}"] = v
                self.results.append(rec)

        # turn into a DataFrame; different rows can have different # of colX,
        # pandas will pad missing ones with NaN
        df = pd.DataFrame(self.results)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info("✅ Saved %d rows to %s", len(df), OUTPUT_CSV)

if __name__ == "__main__":
    TableOCRExtractor(IMAGE_DIR).run()
