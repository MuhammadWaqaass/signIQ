FROM python:3.11-slim

# Install system libraries needed by OpenCV, Pillow, and Supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user according to Hugging Face Spaces standard (UID 1000)
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

# Pre-download weights into container image during build phase (0s runtime cold-start)
COPY --chown=user:user scripts/ scripts/
RUN python scripts/download_models.py

# Copy application code
COPY --chown=user:user . .

# Ensure user permissions for local data and jobs database
RUN mkdir -p /app/data && chown -R user:user /app

USER user

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Start Supervisor to run FastAPI and Celery Worker together
CMD ["supervisord", "-c", "/app/supervisord.conf"]
