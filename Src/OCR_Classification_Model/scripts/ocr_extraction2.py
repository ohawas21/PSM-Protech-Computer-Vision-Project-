import os
import glob
import re
from datetime import datetime

import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import camelot


class OCRProcessor:
    """
    A class to convert images to PDFs, extract text via OCR,
    extract tables via Camelot, and then extract numbers via regex.
    """

    def __init__(self,
                 image_dir,
                 pdf_dir=None,
                 output_csv=None,
                 poppler_path=None):
        """
        :param image_dir: Directory containing input images
        :param pdf_dir: Directory to save generated PDF files
        :param output_csv: Path to save the extracted numbers CSV
        :param poppler_path: Path to Poppler's bin (for Windows)
        """
        self.image_dir = image_dir
        self.pdf_dir = pdf_dir or os.path.join(os.getcwd(), 'pdfs')
        self.output_csv = output_csv or os.path.join(os.getcwd(), 'extracted_numbers.csv')
        self.poppler_path = poppler_path

        os.makedirs(self.pdf_dir, exist_ok=True)

    def convert_images_to_pdf(self):
        """
        Converts all images in image_dir → PDF in pdf_dir. Returns list of PDF paths.
        """
        pdf_paths = []
        for img_path in glob.glob(os.path.join(self.image_dir, '*.*')):
            try:
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                base = os.path.splitext(os.path.basename(img_path))[0]
                pdf_path = os.path.join(self.pdf_dir, f"{base}.pdf")
                img.save(pdf_path, 'PDF', resolution=100.0)
                pdf_paths.append(pdf_path)
            except Exception as e:
                print(f"[convert_images_to_pdf] Failed {img_path}: {e}")
        return pdf_paths

    def extract_tables_from_pdf(self, pdf_path):
        """
        Use Camelot to pull any tables. Returns list of DataFrames.
        """
        dfs = []
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            for table in tables:
                df = table.df.copy()
                df['source_pdf'] = os.path.basename(pdf_path)
                dfs.append(df)
        except Exception as e:
            print(f"[extract_tables_from_pdf] Failed {pdf_path}: {e}")
        return dfs

    def extract_text_from_pdf(self, pdf_path):
        """
        Render each PDF page → image, OCR via pytesseract, return concatenated text.
        """
        text_pages = []
        try:
            if self.poppler_path:
                pages = convert_from_path(pdf_path, poppler_path=self.poppler_path)
            else:
                pages = convert_from_path(pdf_path)

            for page in pages:
                txt = pytesseract.image_to_string(page)
                text_pages.append(txt)
        except Exception as e:
            print(f"[extract_text_from_pdf] Failed {pdf_path}: {e}")

        return "\n".join(text_pages).strip()

    def extract_numbers_and_save(self, regex_patterns=None):
        """
        High-level pipeline:
         1) convert images → PDFs
         2) for each PDF:
             a) extract tables to DF, OCR fallback text
             b) run regex on all text → capture numeric matches
         3) collate into a master CSV of (source, match, pattern)
        """
        # Default patterns: decimals, ints, ± tolerances, diameters, percentages
        default_patterns = {
            'decimal':      r'\b\d+\.\d+\b',
            'integer':      r'\b\d+\b',
            'tolerance_pm': r'±\s*\d+\.\d+',
            'diameter':     r'Ø\s*\d+(\.\d+)?',
            'percent':      r'\d+(\.\d+)?\s*%'}
        patterns = regex_patterns or default_patterns

        pdf_paths = self.convert_images_to_pdf()
        records = []

        for pdf in pdf_paths:
            base = os.path.splitext(os.path.basename(pdf))[0]
            # 1) table text
            tables = self.extract_tables_from_pdf(pdf)
            for df in tables:
                # collapse all cells into one text blob
                all_cells = df.astype(str).agg(' '.join, axis=1).agg(' '.join)
                full_text = all_cells
                for name, pat in patterns.items():
                    for m in re.findall(pat, full_text):
                        records.append({
                            'source': base,
                            'pattern': name,
                            'match': m
                        })

            # 2) OCR fallback text
            ocr_text = self.extract_text_from_pdf(pdf)
            for name, pat in patterns.items():
                for m in re.findall(pat, ocr_text):
                    records.append({
                        'source': base,
                        'pattern': name,
                        'match': m
                    })

        # Save master CSV
        if records:
            df_out = pd.DataFrame(records)
            # add timestamp suffix
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_csv.replace('.csv', f'_{ts}.csv')
            df_out.to_csv(out_path, index=False)
            print(f"[extract_numbers_and_save] Saved {len(records)} matches to {out_path}")
            return out_path
        else:
            print("[extract_numbers_and_save] No numeric matches found.")
            return None

if __name__ == '__main__':
    # === Adjust these paths to your own environment ===
    BASE_IMG_DIR = r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Extraction_Model\Dataset\images"
    POPPLER_BIN = r"C:\poppler-23.05.0\Library\bin"
    OUTPUT_CSV = r"C:\Users\Admin\PSE\extracted_numbers.csv"

    processor = OCRProcessor(
        image_dir=BASE_IMG_DIR,
        poppler_path=POPPLER_BIN,
        output_csv=OUTPUT_CSV
    )
    csv_file = processor.extract_numbers_and_save()
    print("Done. CSV written to:", csv_file)

    

