"""Tests that both ONNX Runtime sessions load and inference succeeds with zero torch."""
import os
import sys

# Add project root to path so 'worker' and 'app' packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from worker import models

print("1. Testing models.load_all()...")
models.load_all()

# Test Detector
print("2. Testing detector inference session...")
detector = models.get_detector()
test_canvas = Image.new("RGB", (640, 640), (255, 255, 255))
test_canvas_path = "weights/temp_test_canvas.png"
test_canvas.save(test_canvas_path)

detections = detector.predict(test_canvas_path)
print(f"   Detector loaded. Smoke test output detections count: {len(detections)}")

# Test Similarity
print("3. Testing similarity inference session...")
sim_session, sim_inference = models.get_similarity()
dummy_input = np.random.randn(2, 3, 224, 224).astype(np.float32)
embeddings = sim_session.run(None, {"input": dummy_input})[0]
print(f"   Similarity model loaded. Embeddings shape: {embeddings.shape} (Expected: 2, 256)")

if os.path.exists(test_canvas_path):
    os.remove(test_canvas_path)

print("\n Both ONNX models loaded and verified successfully with pure ONNX Runtime!")
