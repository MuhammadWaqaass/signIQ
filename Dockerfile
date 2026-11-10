FROM python:3.11-slim

# Install system libraries needed by OpenCV, Pillow, and Supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-create weights directory with correct ownership
RUN mkdir -p /app/weights && chown -R user:user /app/weights

# Copy scripts first (download step — cache busted when scripts/ changes)
COPY --chown=user:user scripts/ scripts/

# Download ONNX weights during build phase for faster startup.
# The entrypoint will re-download if this layer is stale/cached incorrectly.
RUN python scripts/download_models.py

# Copy application code
COPY --chown=user:user . .

# Ensure user permissions for local data and weights
RUN mkdir -p /app/data && chown -R user:user /app

# Make entrypoint executable
RUN chmod +x /app/scripts/entrypoint.sh

USER user

EXPOSE 7860

# Entrypoint checks weights exist before starting supervisord.
# Falls back to downloading if the build-time layer was cached/missed.
CMD ["/app/scripts/entrypoint.sh"]
