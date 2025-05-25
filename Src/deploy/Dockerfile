# Stage 1: Build dependencies and Python packages
FROM python:3.10-slim AS builder

# Install system dependencies needed to build and run the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy app source code
COPY . .

# Create necessary folders and set permissions
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing

# Expose application port
EXPOSE 5001

# Start the Flask app (use Gunicorn in production if needed)
CMD ["python", "app.py"]