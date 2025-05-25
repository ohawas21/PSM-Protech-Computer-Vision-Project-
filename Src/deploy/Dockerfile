# Stage 1: Build Python dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy and install Python dependencies
# System dependencies required for pip install (e.g., gcc, python3-dev) would go here if needed.
# For the current requirements.txt, these are not typically needed for the install phase itself.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime image
FROM python:3.10-slim

# Install system-level runtime dependencies required by the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
# Copy Python executable and other necessary parts of the Python installation if needed,
# though usually site-packages is sufficient if base images match.
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy all source code
COPY . .

# ✅ Explicitly copy model weights into /app/models/
# This ensures models are in the correct location even if they are in a subfolder of the build context.
COPY models/ /app/models/

# Prepare working directories with appropriate permissions
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 /app/models # Ensure models are readable

# Expose Flask port
EXPOSE 5001

# Run the application (development mode; replace with gunicorn for production)
CMD ["python", "app.py"]