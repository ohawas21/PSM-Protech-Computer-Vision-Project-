# OCR Pipeline Change Log

All notable changes to the OCR Pipeline are recorded here, with timestamps, descriptions, and rationale.

## Unreleased

- **2025-04-26 22:39** Add `convert_images_to_pdf` method (initial step)  
  _commit: feat: add convert_images_to_pdf_  
  - **What it does:** Scans the specified image directory, opens each file with PIL, converts it to RGB, and saves it as a PDF into an in-memory buffer.  
  - **Why it’s necessary:** Standardizes all inputs as PDFs without touching disk, enabling faster downstream PDF-based processing and simplifying cleanup.

- **2025-04-26 22:39** Add `render_pdfs_to_images` method (render PDFs back to images)  
  _commit: feat: add render_pdfs_to_images_  
  - **What it does:** Uses `pdf2image.convert_from_bytes` to render each in-memory PDF buffer at 300 DPI and stores the first page as a PIL image.  
  - **Why it’s necessary:** Ensures we work with a consistent image representation for cropping and OCR, bridging the PDF buffer step to image-based processing.

- **2025-04-26 22:39** Add `crop_regions` method (crop page images into Symbol, Value1, Value2)  
  _commit: feat: add crop_regions step_  
  - **What it does:** Calculates predefined column boundaries (12%, 52%, 100% of width) on each rendered page and crops out three regions labeled Symbol, Value1, and Value2.  
  - **Why it’s necessary:** Isolates individual table columns to improve OCR accuracy by focusing Tesseract on specific data fields rather than the whole page.

- **2025-04-26 22:39** Initialize LaTeX change log file  
  _commit: docs: add OCRPIPELINE_CHANGELOG.tex_  
  - **What it does:** Creates a professor-friendly LaTeX document mirroring this changelog.  
  - **Why it’s necessary:** Meets requirement for formal documentation in LaTeX, allowing clean typeset output for academic review.

*This file tracks the evolution of the OCR pipeline in a human-readable format.*
