# Stage 1: Build with Python and required system libraries
FROM python:3.10-slim AS builder

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Minimal runtime container
FROM python:3.10-slim

# Reinstall runtime libraries (needed for OpenCV, Tesseract, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy application code
COPY . .

# Create necessary working directories
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing

# Expose port for Flask
EXPOSE 5001

# Start Flask app (or Gunicorn in production)
CMD ["python", "app.py"]