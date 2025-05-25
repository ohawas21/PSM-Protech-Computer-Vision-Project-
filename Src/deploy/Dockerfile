# Dockerfile for the Document Processing Pipeline

# Stage 1: Base image with Python and build tools
FROM python:3.11-slim-bullseye AS builder

# Install Tesseract OCR, libmagic, and other dependencies
# poppler-utils is for pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    # Add any other system-level dependencies your project might need
    # For example, if OpenCV needs more, or for specific font rendering for tesseract
    # libgl1-mesa-glx libglib2.0-0 
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim-bullseye

# Set TESSDATA_PREFIX for Tesseract if needed (often handled by tesseract install)
# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# Copy installed packages and dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/bin/tesseract /usr/bin/tesseract
COPY --from=builder /usr/share/tesseract-ocr /usr/share/tesseract-ocr
COPY --from=builder /usr/bin/pdftoppm /usr/bin/pdftoppm 
COPY --from=builder /usr/bin/pdfinfo /usr/bin/pdfinfo   
# If python-magic or other libs need their shared libraries explicitly copied:
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmagic.so.1 /usr/lib/x86_64-linux-gnu/libmagic.so.1
# Add other necessary binaries/libraries if discovered during testing

# Copy application code
COPY . .

# Create necessary directories and set permissions if needed
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing \
    && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing

# Expose port
EXPOSE 5001

# Command to run the application
# Using Gunicorn is recommended for production, but for simplicity, Flask dev server is used here.
# For Gunicorn: CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
CMD ["python", "app.py"]
