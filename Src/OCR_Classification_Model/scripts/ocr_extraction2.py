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
