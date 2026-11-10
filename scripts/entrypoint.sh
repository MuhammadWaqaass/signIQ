#!/bin/bash
# Entrypoint: ensure ONNX weights exist before starting services.
# If the Docker build layer was cached and weights are missing, download now.
set -e

WEIGHTS_DIR="/app/weights"
REQUIRED_FILES=("best.onnx" "similarity.onnx" "similarity.onnx.data")

echo "[entrypoint] Checking ONNX weights in ${WEIGHTS_DIR}..."

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${WEIGHTS_DIR}/${f}" ] || [ ! -s "${WEIGHTS_DIR}/${f}" ]; then
        echo "[entrypoint] Missing or empty: ${f}"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "[entrypoint] Downloading missing ONNX models from Hugging Face..."
    python /app/scripts/download_models.py
    echo "[entrypoint] Download complete."
else
    echo "[entrypoint] All weights present. Skipping download."
fi

exec supervisord -c /app/supervisord.conf
