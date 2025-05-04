<<<<<<< HEAD
import os
import glob
import re
from PIL import Image
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import camelot

class OCRProcessor:
    """
    A class to convert images to PDF, extract text via OCR, extract tables via Camelot, and clean or append the data.
    """
    def __init__(self, image_dir, pdf_dir=None, output_csv=None, poppler_path=None):
        """
        :param image_dir: Directory containing input images
        :param pdf_dir: Directory to save generated PDF files (defaults to './pdfs')
        :param output_csv: Path to save the extracted data CSV (defaults to './extracted_data.csv')
        :param poppler_path: Optional filesystem path to Poppler's bin directory
        """
        self.image_dir = image_dir
        self.pdf_dir = pdf_dir or os.path.join(os.getcwd(), 'pdfs')
        self.output_csv = output_csv or os.path.join(os.getcwd(), 'extracted_data.csv')
        self.poppler_path = poppler_path
        os.makedirs(self.pdf_dir, exist_ok=True)

    def convert_images_to_pdf(self):
        """
        Converts all images in `self.image_dir` to PDF files in `self.pdf_dir`.
        Returns a list of the PDF file paths.
        """
        image_paths = glob.glob(os.path.join(self.image_dir, '*.*'))
        pdf_paths = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                base = os.path.splitext(os.path.basename(img_path))[0]
                pdf_path = os.path.join(self.pdf_dir, f"{base}.pdf")
                img.save(pdf_path, 'PDF', resolution=100.0)
                pdf_paths.append(pdf_path)
            except Exception as e:
                print(f"Failed to convert {img_path}: {e}")
        return pdf_paths

    def extract_tables_from_pdf(self, pdf_path):
        """
        Uses Camelot to extract all tables from a PDF file.
        Returns a list of DataFrames (one per table).
        """
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            return [table.df for table in tables]
        except Exception as e:
            print(f"Failed to extract tables from {pdf_path}: {e}")
            return []

    def process_folder_append_csv(self):
        """
        High-level method: for each image in `self.image_dir`:
          - convert to PDF
          - extract tables via Camelot
          - OCR the pages for fallback text
          - append both table rows and OCR text to `self.output_csv`
        Returns the CSV path.
        """
        records = []
        image_paths = glob.glob(os.path.join(self.image_dir, '*.*'))

        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            pdf_path = os.path.join(self.pdf_dir, f"{base}.pdf")
            try:
                # convert image to PDF
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                img.save(pdf_path, 'PDF', resolution=100.0)

                # extract tables
                tables = self.extract_tables_from_pdf(pdf_path)
                for idx, df_table in enumerate(tables):
                    df_table['source_file'] = base
                    df_table['table_index'] = idx
                    # flatten multi-row header if necessary
                    df_flat = df_table.copy()
                    records.extend(df_flat.to_dict(orient='records'))

                # fallback OCR
                if self.poppler_path:
                    pages = convert_from_path(pdf_path, poppler_path=self.poppler_path)
                else:
                    pages = convert_from_path(pdf_path)
                ocr_text = '\n'.join(pytesseract.image_to_string(page) for page in pages).strip()
                if ocr_text:
                    records.append({'source_file': base, 'table_index': None, 'text': ocr_text})

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if records:
            df_combined = pd.DataFrame(records)
            if os.path.exists(self.output_csv):
                try:
                    df_existing = pd.read_csv(self.output_csv)
                    df_combined = pd.concat([df_existing, df_combined], ignore_index=True)
                except pd.errors.EmptyDataError:
                    pass
            df_combined.to_csv(self.output_csv, index=False)
            print(f"Appended data from {len(image_paths)} files to {self.output_csv}")
        else:
            print("No data extracted; CSV unchanged.")

        return self.output_csv

# Example usage:
if __name__ == '__main__':
    BASE = r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\Dataset\final_annotation"
    processor = OCRProcessor(
        image_dir=BASE,
        poppler_path=r"C:\poppler-23.05.0\Library\bin"
    )
    csv_out = processor.process_folder_append_csv()
    print(f"Data aggregated in: {csv_out}")

=======
import os
import glob
import re
from PIL import Image
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import camelot

class OCRProcessor:
    """
    A class to convert images to PDF, extract text via OCR, extract tables via Camelot, and clean or append the data.
    """
    def __init__(self, image_dir, pdf_dir=None, output_csv=None, poppler_path=None):
        """
        :param image_dir: Directory containing input images
        :param pdf_dir: Directory to save generated PDF files (defaults to './pdfs')
        :param output_csv: Path to save the extracted data CSV (defaults to './extracted_data.csv')
        :param poppler_path: Optional filesystem path to Poppler's bin directory
        """
        self.image_dir = image_dir
        self.pdf_dir = pdf_dir or os.path.join(os.getcwd(), 'pdfs')
        self.output_csv = output_csv or os.path.join(os.getcwd(), 'extracted_data.csv')
        self.poppler_path = poppler_path
        os.makedirs(self.pdf_dir, exist_ok=True)

    def convert_images_to_pdf(self):
        """
        Converts all images in `self.image_dir` to PDF files in `self.pdf_dir`.
        Returns a list of the PDF file paths.
        """
        image_paths = glob.glob(os.path.join(self.image_dir, '*.*'))
        pdf_paths = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                base = os.path.splitext(os.path.basename(img_path))[0]
                pdf_path = os.path.join(self.pdf_dir, f"{base}.pdf")
                img.save(pdf_path, 'PDF', resolution=100.0)
                pdf_paths.append(pdf_path)
            except Exception as e:
                print(f"Failed to convert {img_path}: {e}")
        return pdf_paths

    def extract_tables_from_pdf(self, pdf_path):
        """
        Uses Camelot to extract all tables from a PDF file.
        Returns a list of DataFrames (one per table).
        """
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            return [table.df for table in tables]
        except Exception as e:
            print(f"Failed to extract tables from {pdf_path}: {e}")
            return []

    def process_folder_append_csv(self):
        """
        High-level method: for each image in `self.image_dir`:
          - convert to PDF
          - extract tables via Camelot
          - OCR the pages for fallback text
          - append both table rows and OCR text to `self.output_csv`
        Returns the CSV path.
        """
        records = []
        image_paths = glob.glob(os.path.join(self.image_dir, '*.*'))

        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            pdf_path = os.path.join(self.pdf_dir, f"{base}.pdf")
            try:
                # convert image to PDF
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                img.save(pdf_path, 'PDF', resolution=100.0)

                # extract tables
                tables = self.extract_tables_from_pdf(pdf_path)
                for idx, df_table in enumerate(tables):
                    df_table['source_file'] = base
                    df_table['table_index'] = idx
                    # flatten multi-row header if necessary
                    df_flat = df_table.copy()
                    records.extend(df_flat.to_dict(orient='records'))

                # fallback OCR
                if self.poppler_path:
                    pages = convert_from_path(pdf_path, poppler_path=self.poppler_path)
                else:
                    pages = convert_from_path(pdf_path)
                ocr_text = '\n'.join(pytesseract.image_to_string(page) for page in pages).strip()
                if ocr_text:
                    records.append({'source_file': base, 'table_index': None, 'text': ocr_text})

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if records:
            df_combined = pd.DataFrame(records)
            if os.path.exists(self.output_csv):
                try:
                    df_existing = pd.read_csv(self.output_csv)
                    df_combined = pd.concat([df_existing, df_combined], ignore_index=True)
                except pd.errors.EmptyDataError:
                    pass
            df_combined.to_csv(self.output_csv, index=False)
            print(f"Appended data from {len(image_paths)} files to {self.output_csv}")
        else:
            print("No data extracted; CSV unchanged.")

        return self.output_csv

# Example usage:
if __name__ == '__main__':
    BASE = r"C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\Dataset\final_annotation"
    processor = OCRProcessor(
        image_dir=BASE,
        poppler_path=r"C:\poppler-23.05.0\Library\bin"
    )
    csv_out = processor.process_folder_append_csv()
    print(f"Data aggregated in: {csv_out}")

>>>>>>> dac556b2e3a093c229aca5c71cdbb7081c327a13
