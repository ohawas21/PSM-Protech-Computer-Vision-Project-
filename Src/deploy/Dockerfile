# Stage 1: Build Python dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime image
FROM python:3.10-slim

WORKDIR /app

# Install system-level runtime dependencies required by OpenCV, Tesseract, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source code and models
COPY . .
COPY models/ /app/models/

# Embed build version
ARG BUILD_TIMESTAMP
LABEL build_time=$BUILD_TIMESTAMP

# Prepare runtime directories
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 /app/models

# Expose Flask port
EXPOSE 5001

# Start the application using Gunicorn (for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]