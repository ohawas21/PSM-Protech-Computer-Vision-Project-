# File: Src/OCR_Extraction_Model/Scripts/ocr_pipeline.py
import os
import re
import csv
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict

class OCRPipeline:
    def __init__(self, tesseract_cmd: str = None):
        """
        Initialize the pipeline. If you have a custom tesseract path, pass it here.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def images_to_pdf(self, image_folder: str, output_pdf: str):
        """
        Method 1 (already done): convert all images in image_folder to a single PDF.
        """
        from PIL import Image

        images = []
        for fname in sorted(os.listdir(image_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                path = os.path.join(image_folder, fname)
                images.append(Image.open(path).convert('RGB'))

        if not images:
            raise FileNotFoundError(f"No images found in {image_folder}.")

        # Save all images as a multipage PDF
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"[INFO] Saved PDF: {output_pdf}")

    def pdf_to_text(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """
        Method 2: Render each PDF page to image and run OCR to get text.
        Returns a list of page_text strings.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"{pdf_path} not found.")

        # Convert PDF pages to PIL images
        pages = convert_from_path(pdf_path, dpi=dpi)
        page_texts = []
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            page_texts.append(text)
            print(f"[DEBUG] OCR page {i+1}/{len(pages)} length: {len(text)} chars")

        return page_texts

    def extract_numbers(self, texts: List[str], pattern: str = r"[-+]?\d*\.\d+|\d+") -> List[List[str]]:
        """
        Method 3a: From each page's text, extract all numbers matching the regex pattern.
        Returns a list of lists: one sublist per page.
        Default pattern captures floats and ints.
        """
        compiled = re.compile(pattern)
        all_numbers = []
        for i, text in enumerate(texts):
            nums = compiled.findall(text)
            print(f"[DEBUG] Found {len(nums)} matches on page {i+1}")
            all_numbers.append(nums)
        return all_numbers

    def save_numbers_csv(self,
                         numbers_per_page: List[List[str]],
                         output_csv: str,
                         include_page: bool = True):
        """
        Method 3b: Save the extracted numbers to a CSV file.
        If include_page=True, CSV columns: page, number_index, value
        """
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["page", "index", "value"] if include_page else ["value"]
            writer.writerow(header)

            for page_idx, nums in enumerate(numbers_per_page, start=1):
                for idx, val in enumerate(nums, start=1):
                    if include_page:
                        writer.writerow([page_idx, idx, val])
                    else:
                        writer.writerow([val])

        print(f"[INFO] Saved extracted numbers to CSV: {output_csv}")

if __name__ == "__main__":
    # Example usage
    pipeline = OCRPipeline()

    # 1) Images → PDF
    img_dir = "test_images"
    out_pdf = "output/combined.pdf"
    pipeline.images_to_pdf(img_dir, out_pdf)

    # 2) OCR PDF → raw text
    page_texts = pipeline.pdf_to_text(out_pdf)

    # 3a) Extract numbers via regex
    numbers = pipeline.extract_numbers(page_texts,
                                       pattern=r"[-+]?\d*\.\d+|\d+")

    # 3b) Save to CSV
    pipeline.save_numbers_csv(numbers, "output/extracted_numbers.csv")