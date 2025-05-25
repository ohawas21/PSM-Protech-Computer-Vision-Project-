# Stage 1: Install Python packages and system dependencies
FROM python:3.10-slim AS builder

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime image
FROM python:3.10-slim

# Reinstall system-level runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python dependencies
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy all source code
COPY . .

# ✅ Explicitly copy model weights into /app/models/
COPY models/ /app/models/

# Prepare working directories with appropriate permissions
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing

# Expose Flask port
EXPOSE 5001

# Run the application (development mode; replace with gunicorn for production)
CMD ["python", "app.py"]