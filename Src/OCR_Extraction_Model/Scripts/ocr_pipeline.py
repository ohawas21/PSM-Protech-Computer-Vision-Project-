import logging
from pathlib import Path
from io import BytesIO
import re
import csv
import argparse

from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

# Configure logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class OCRPipeline:
    """
    An in-memory OCR pipeline that converts images to PDFs, extracts table data,
    and writes per-image and combined CSVs without saving PDFs to disk.

    Usage (from project root):
        python Src/OCR_Extraction_Model/Scripts/ocr_pipeline.py \
            --image_dir Src/OCR_Extraction_Model/Dataset \
            --per_image_dir Src/csvs \
            --combined_csv_path Src/OCR_Extraction_Model/combined.csv \
            [--poppler_path C:/path/to/poppler/bin] \
            [--tesseract_cmd "C:/Program Files/Tesseract-OCR/tesseract.exe"]
    """
    def __init__(self, image_dir: str, per_image_dir: str, combined_csv_path: str, poppler_path: str = None):
        self.image_dir = Path(image_dir)
        self.per_image_dir = Path(per_image_dir)
        self.combined_csv_path = Path(combined_csv_path)
        self.poppler_path = poppler_path
        self._pdf_buffers: dict[str, BytesIO] = {}
        self._extracted_rows: list[dict[str, str]] = []

    def convert_images_to_pdf(self):
        """Convert images in the directory into in-memory PDF buffers."""
        logging.info(f"Looking for images in: {self.image_dir.resolve()}")
        imgs = [p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}]
        logging.info(f"[STEP 1] Converting {len(imgs)} images to PDF buffers…")
        for img_path in imgs:
            logging.info(f"  • {img_path.name}")
            try:
                img = Image.open(img_path).convert('RGB')
                buf = BytesIO()
                img.save(buf, format='PDF')
                buf.seek(0)
                self._pdf_buffers[img_path.stem] = buf
            except Exception as e:
                logging.warning(f"Failed to convert {img_path.name}: {e}")

    def extract_table(self):
        """
        Render each in-memory PDF back to an image, split into three custom columns,
        preprocess each region, debug-save the slice, OCR with a whitelist, and store the raw text.
        """
        total = len(self._pdf_buffers)
        logging.info(f"[STEP 2] Extracting OCR from {total} PDF buffers…")
        for idx, (name, pdf_buf) in enumerate(self._pdf_buffers.items(), start=1):
            logging.info(f"  • ({idx}/{total}) Processing '{name}'")
            try:
                pages = convert_from_bytes(pdf_buf.getvalue(), dpi=300, poppler_path=self.poppler_path)
                if not pages:
                    logging.warning(f"No pages returned for {name}")
                    continue
                page_img = pages[0]
                w, h = page_img.size
                # Custom column bounds (tuned for angularity table layout)
                # Adjust fractions based on debug crops: symbol ~12%, val1 ~40%, val2 ~48%
                bounds = [
                    (0,            int(w * 0.12)),   # narrow ∠ cell (~12% width)
                    (int(w * 0.12), int(w * 0.52)),  # first numeric cell (~40% width)
                    (int(w * 0.52), w)               # second numeric cell (~48% width)
                ]
                row = {"file": name}
                for i, col_name in enumerate(["Symbol", "Value1", "Value2"]):
                    left, right = bounds[i]
                    region = page_img.crop((left, 0, right, h))
                    # DEBUG: save crop to inspect correctness
                    region.save(f"debug_{name}_{col_name}.png")
                    # Preprocess: grayscale, binarize, upscale
                    gray = region.convert("L")
                    bw   = gray.point(lambda x: 0 if x < 128 else 255, '1')
                    up   = bw.resize((gray.width * 2, gray.height * 2), Image.NEAREST)
                    # OCR with whitelist
                    config = "--psm 6 -c tessedit_char_whitelist=0123456789.∠"
                    raw = pytesseract.image_to_string(up, config=config)
                    logging.info(f"    [DEBUG] {name}:{col_name} → '{raw.strip()}'")
                    row[col_name] = raw.strip()
                self._extracted_rows.append(row)
            except Exception as e:
                logging.warning(f"OCR failed for {name}: {e}")

    def clean_and_export_csv(self):
        """Apply regex cleaning and write per-image and combined CSVs."""
        logging.info("[STEP 3] Cleaning text and writing CSV files…")
        self.per_image_dir.mkdir(parents=True, exist_ok=True)
        self.combined_csv_path.parent.mkdir(parents=True, exist_ok=True)

        grouped: dict[str, list] = {}
        for r in self._extracted_rows:
            grouped.setdefault(r['file'], []).append(r)

        header = ["file", "Symbol", "Value1", "Value2"]
        # Write per-image CSVs
        for fname, rows in grouped.items():
            out_path = self.per_image_dir / f"{fname}.csv"
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for r in rows:
                    cleaned = [
                        re.sub(r"[^A-Za-z0-9\.\- ]", "", re.sub(r"\s+", " ", r[c]).strip())
                        for c in header
                    ]
                    writer.writerow(cleaned)
            logging.info(f"  • Wrote per-image CSV: {out_path}")

        # Write combined CSV
        with open(self.combined_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for rows in grouped.values():
                for r in rows:
                    cleaned = [
                        re.sub(r"[^A-Za-z0-9\.\- ]", "", re.sub(r"\s+", " ", r[c]).strip())
                        for c in header
                    ]
                    writer.writerow(cleaned)
        logging.info(
            f"  • Wrote combined CSV: {self.combined_csv_path} ({len(self._extracted_rows)} rows)"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCRPipeline.")
    parser.add_argument("--image_dir", default="Src/OCR_Extraction_Model/Dataset")
    parser.add_argument("--per_image_dir", default="Src/csvs")
    parser.add_argument("--combined_csv_path", default="Src/OCR_Extraction_Model/combined.csv")
    parser.add_argument("--poppler_path", default=None)
    parser.add_argument("--tesseract_cmd", default=None,
                        help="Full path to tesseract executable, if not in PATH")
    args = parser.parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    pipeline = OCRPipeline(
        image_dir=args.image_dir,
        per_image_dir=args.per_image_dir,
        combined_csv_path=args.combined_csv_path,
        poppler_path=args.poppler_path
    )
    pipeline.convert_images_to_pdf()
    pipeline.extract_table()
    pipeline.clean_and_export_csv()
