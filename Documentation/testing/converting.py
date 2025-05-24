#!/usr/bin/env python3
"""
Script: pdf_to_png_converter.py

Description:
    Converts all PDF files in a specified input directory to PNG images,
    saving each page as a separate PNG file in the output directory.

Dependencies:
    - pdf2image (pip install pdf2image)
    - poppler-utils (system package; e.g., apt install poppler-utils)

Usage:
    Edit the `input_dir` and `output_dir` variables under `if __name__ == '__main__'`,
    then run:
        python3 pdf_to_png_converter.py
"""
import os
from pathlib import Path
from pdf2image import convert_from_path

def convert_pdfs_to_pngs(input_dir: str, output_dir: str, dpi: int = 200) -> None:
    """
    Converts all PDF files in `input_dir` to PNG images stored in `output_dir`.

    Args:
        input_dir (str): Relative or absolute path to the directory containing PDF files.
        output_dir (str): Relative or absolute path where PNGs will be saved.
        dpi (int): Resolution in dots per inch for the output images.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all PDF files in the input directory
    pdf_files = list(in_path.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {in_path}")
        return

    for pdf_file in pdf_files:
        # Convert each page of the PDF to an image
        pages = convert_from_path(str(pdf_file), dpi=dpi)
        base_name = pdf_file.stem

        for page_number, page in enumerate(pages, start=1):
            png_filename = f"{base_name}_page_{page_number}.png"
            output_file = out_path / png_filename
            page.save(output_file, "PNG")
            print(f"Saved {output_file}")

    print(f"Finished converting {len(pdf_files)} PDF(s) to PNG(s) in {out_path}")

if __name__ == "__main__":
    # TODO: Set the input and output paths below
    input_dir = "path/to/your/input_directory"
    output_dir = "path/to/your/output_directory"
    
    convert_pdfs_to_pngs(input_dir, output_dir)
