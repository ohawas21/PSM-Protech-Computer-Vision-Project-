# Stage 2: Final lightweight runtime image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy source code
COPY . .
COPY models/ /app/models/

# Build metadata
ARG BUILD_TIMESTAMP
LABEL build_time=$BUILD_TIMESTAMP

# Prepare runtime folders
RUN mkdir -p Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 Falcon_r1_preprocess Falcon_r2_preprocess Falcon_r3_input outputs results temp_processing \
 && chmod -R 755 /app/models

EXPOSE 5001

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]