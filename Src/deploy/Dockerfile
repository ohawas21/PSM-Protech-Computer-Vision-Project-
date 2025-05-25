# Stage 1: Install Python packages and system dependencies
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime container
FROM python:3.10-slim

# ✅ This step is **CRITICAL** to make sure OpenCV works
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy application source
COPY . .

# Prepare directories
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input models outputs results temp_processing

EXPOSE 5001

CMD ["python", "app.py"]