# Stage 1: Build Python dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy source
COPY . .
COPY models/ /app/models/

# Build-time metadata
ARG BUILD_TIMESTAMP
LABEL build_time=$BUILD_TIMESTAMP

# Prepare folders
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 /app/models

EXPOSE 5001

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
