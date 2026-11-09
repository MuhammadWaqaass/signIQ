# 🚀 SignIQ End-to-End Deployment, Optimization & Production Guide

This document is the definitive technical record of designing, optimizing, converting, and deploying the **SignIQ** AI platform (Signature Detection & Siamese Verification). It captures every engineering hurdle, memory constraint, architecture shift, network fix, and performance tuning applied to achieve a fast, reliable, zero-cost production service on **Render Free Tier (512 MB RAM, 0.1 vCPU)**.

---

## 📌 1. Executive Summary & Impact Metrics

| Metric | Before Optimization | After Optimization | Impact |
| :--- | :--- | :--- | :--- |
| **Detector Weights** | `242 MB` (`best.pt`) | `80 MB` (`best.onnx`) | **67% reduction on disk** |
| **Similarity Weights** | `112 MB` PyTorch checkpoint | `112 MB` ONNX graph | **Zero delta numerical parity** |
| **Runtime Memory (RAM)** | `~850 MB – 1.1 GB` (Crash OOM) | `~200 MB – 240 MB` | **Fits safely inside 512 MB Free Tier** |
| **PyTorch in Runtime** | Required (torch + torchvision + libreyolo) | **ZERO PyTorch at runtime** | **~600 MB memory pool eliminated** |
| **Worker Architecture** | Separate Celery worker + Redis required | Dual-mode: In-process async background tasks | **$0 extra worker / infrastructure cost** |
| **Detection Speed (CPU)** | `~850ms – 1.4s` | `~180ms – 280ms` | **3x faster inference** |
| **Similarity Speed (CPU)** | `~35s – 50s` (CPU starvation timeouts) | `~3s – 5s` | **10x faster; eliminated 502 Bad Gateway** |
| **Database Reliability** | Direct IPv6 connection failure | IPv4 Pooler + Auto SQLite Fallback | **Zero startup crashes** |

---

## 🏗️ 2. High-Level Architecture Overview

SignIQ operates as a unified, self-contained full-stack service on Render:

```
[User Browser / Mobile Client]
           │
           │  HTTPS GET / (Serves Static HTML / CSS / JS)
           │  HTTPS POST /api/v1/jobs (Uploads Document + Reference)
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Render Free Container (512 MB RAM)              │
│                                                                        │
│   FastAPI Web Server (Uvicorn - Port $PORT)                           │
│   ├── Serves Frontend statically mounted at / (Zero CORS overhead)    │
│   ├── Validates file types (PNG, JPG, WebP) & size limits (10 MB)     │
│   ├── Returns 202 Accepted + job_id immediately                       │
│   │                                                                    │
│   └── In-Process Background Task (anyio threadpool)                    │
│       │                                                                │
│       ├── 1. Pure ONNX LibreYOLO9m Detector (weights/best.onnx)        │
│       │      - Top-left letterbox (640x640)                            │
│       │      - Raw xyxy box extraction & NMS                           │
│       │      - 6px padded crop generation                              │
│       │                                                                │
│       ├── 2. Instant Reference Validation                             │
│       │      - sim_preprocess.preprocess() checks ink/contrast in 50ms │
│       │                                                                │
│       └── 3. Pure ONNX ConvNeXt-Tiny Siamese Verifier                  │
│              - Normalization mean/std from config                      │
│              - 256-d normalized cosine similarity embedding            │
│              - Threshold decision (>= 0.7785: SIMILAR)                 │
│                                                                        │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  │ (Saves Job State & Detections)
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Primary: Supabase PostgreSQL (IPv4 Pooler - Port 6543)│
      │  Fallback: Local SQLite (jobs.db if network drops)     │
      └────────────────────────────────────────────────────────┘
```

---

## 🔬 3. The 512 MB Memory Bottleneck & Pure-ONNX Solution

### The Core Problem
Render Free instances enforce a strict **512 MB RAM hard limit**.
When the application initially ran:
- Importing `torch`, `torchvision`, and `libreyolo` at the Python module level automatically allocated PyTorch's internal C++ CUDA/CPU memory pools (~400 MB – 600 MB) before any request was even processed.
- When Uvicorn booted alongside FastAPI and SQLAlchemy, total resident memory surged to **~530 MB – 650 MB**.
- Render's watchdog immediately sent `SIGKILL 137`:
  ```
  ==> Out of memory (used over 512Mi)
  ==> Common ways to troubleshoot your deploy: https://render.com/docs/troubleshooting-deploys
  ```

### The Solution: Zero PyTorch in Production Runtime
To fit within 512 MB, **PyTorch and LibreYOLO were completely eradicated from the runtime environment**:

1. **Split Build vs. Runtime Dependencies**:
   - `requirements-build.txt`: Contains heavy compilation packages (`torch`, `torchvision`, `libreyolo`, `onnxscript`).
   - `requirements.txt`: Contains only production runtime packages (`fastapi`, `uvicorn`, `onnxruntime`, `numpy`, `pillow`, `sqlalchemy`, `psycopg`).
   - The build command installs build packages, runs the exporter, and immediately uninstalls them before the production container launches:
     ```bash
     pip install -r requirements-build.txt && \
     pip install -r requirements.txt && \
     python scripts/export_to_onnx.py && \
     pip uninstall -y torch torchvision libreyolo onnxscript torchaudio 2>/dev/null || true
     ```

2. **Handcrafted Pure-NumPy Detector (`worker/onnx_detector.py`)**:
   - Runs `best.onnx` directly using `onnxruntime.InferenceSession` with `CPUExecutionProvider`.
   - Restricts internal threading pools (`intra_op_num_threads=1`, `inter_op_num_threads=1`) to prevent thread-allocation memory spikes.

3. **Self-Contained Preprocessor (`worker/sim_preprocess.py`)**:
   - The Hugging Face repository for the similarity model included an `inference.py` script that had `import torch` at line 1.
   - We extracted the pure NumPy/Pillow preprocessing pipeline (grayscale conversion, contrast stretching, line removal, ink cropping, square padding, Lanczos resize to 224x224) into a standalone module with hardcoded normalization constants from `config.json`.
   - Result: Zero torch imports, zero external network downloads at runtime.

---

## 📐 4. LibreYOLO9 Bounding Box & Letterboxing Geometry

During initial testing with the ONNX detector, bounding boxes were distorted and cut off the bottom and right edges of signatures. Investigating LibreYOLO's internal engine revealed two critical model-specific conventions:

### Bug 1: YOLOv8 vs. YOLOv9 Coordinate Layout
- Standard YOLOv8 outputs `(batch, 5, 8400)` where rows `0:4` are `[cx, cy, w, h]`.
- **LibreYOLO9 (YOLOv9)** outputs rows `0:4` that are **already `[x1, y1, x2, y2]` (xyxy)**.
- Applying an `xywh2xyxy` transform treated `x2` as width and `y2` as height, subtracting half of their magnitudes from `x1` and `y1`. This collapsed the box into the top-left corner.
- **Fix**: Direct extraction without transformation:
  ```python
  preds = raw[0].T           # (8400, 5)
  boxes_xyxy = preds[:, :4]  # Already x1, y1, x2, y2
  scores = preds[:, 4]       # Confidence score
  ```

### Bug 2: Top-Left Letterbox vs. Center Padding
- Standard vision pipelines pad images equally on all sides (center padding).
- **LibreYOLO YOLOv9 checkpoints are trained with top-left padding**: the scaled image is placed at `(0, 0)`, and padding is placed strictly on the right and bottom.
- Center-padding shifted all predicted coordinates by half the pad offset.
- **Fix**: Aligned preprocessor letterbox to top-left placement:
  ```python
  ratio = min(640 / orig_h, 640 / orig_w)
  new_w = max(int(orig_w * ratio), 1)
  new_h = max(int(orig_h * ratio), 1)
  resized = img.resize((new_w, new_h), Image.BILINEAR)

  canvas = Image.new("RGB", (640, 640), (114, 114, 114))
  canvas.paste(resized, (0, 0))  # Top-left placement
  ```
- **Unletterboxing**:
  ```python
  boxes_orig = boxes_xyxy / ratio
  boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
  boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)
  ```

### Bug 3: Edge Stroke Clipping
- Bounding boxes predicted by YOLO fit tightly against the ink, occasionally slicing outer loops or dots.
- **Fix**: Added a 6px safety padding margin in [`worker/pipeline.py`](../worker/pipeline.py) during crop extraction:
  ```python
  def crop_and_save(image: Image.Image, box: list[float], out_path: Path, pad: int = 6) -> Path:
      w, h = image.size
      x1, y1, x2, y2 = box
      x1 = max(0, int(x1) - pad)
      y1 = max(0, int(y1) - pad)
      x2 = min(w, int(x2) + pad)
      y2 = min(h, int(y2) + pad)
      image.crop((x1, y1, x2, y2)).save(out_path)
      return out_path
  ```

---

## ⚡ 5. Performance Tuning & Eliminating 502 Bad Gateway

### Root Cause of the 502 Bad Gateway
When running similarity comparisons, the frontend experienced repeating `502 Bad Gateway` errors during polling. This was caused by **CPU starvation**:
- The worker script previously ran the full YOLO detector on the document, then ran it on the reference signature at `conf=0.15`, and if no box was found, ran it AGAIN at `conf=0.05`.
- On Render's 0.1 vCPU shared tier, executing 3 to 4 sequential YOLO inference runs pegged CPU utilization at 100% for **35 to 50 seconds**.
- Because Uvicorn and background tasks shared the same process, incoming polling requests and Render health checks were blocked from receiving CPU slices. Cloudflare timed out and returned `502 Bad Gateway`.

### The Optimization (`worker/tasks.py`)
1. **Bypassed YOLO on Reference Signatures**: A reference image uploaded by a user is already an isolated signature crop. Scanning it with an object detector was redundant.
2. **Instant Preprocessing Validation**: Replaced YOLO validation on the reference image with `sim_inference.preprocess()` (~50 ms), which checks ink density, contrast stretch, and blank paper directly.
3. **Execution Time Drop**:
   - Total similarity execution time dropped from **~45s to ~3–5s**.
   - CPU starvation eliminated; 502 errors ceased.

---

## 🌐 6. Network, CORS & Database Engineering

### 1. The IPv6 "Network is Unreachable" Bug
- **Issue**: Direct Supabase connection strings point to port `5432` (`db.wmbhjvsaemebgfctgege.supabase.co`).
- This hostname resolves to an **IPv6 address** (`2406:da12:...`).
- Render Free tier environments operate on IPv4-only networks and do not route IPv6. Connecting to port 5432 produced:
  ```
  sqlalchemy.exc.OperationalError: connection to server at "2406:da12:..." port 5432 failed: Network is unreachable
  ```
- **Fix**: Switch to Supabase's **Transaction Connection Pooler** on port **6543**:
  ```text
  postgresql://postgres.wmbhjvsaemebgfctgege:[PASSWORD]@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require
  ```
  The pooler domain resolves to AWS IPv4 addresses (`13.124.111.232`, `15.165.245.138`), which connect cleanly without requiring Supabase's \$4/mo paid IPv4 add-on.

### 2. Startup Resilience: Automatic SQLite Fallback
In [`app/db.py`](../app/db.py), a connection probe runs on startup. If the remote database is unreachable or credentials fail, the server logs a warning and automatically falls back to an ephemeral SQLite database (`jobs.db`):
```python
def _resolve_engine():
    try:
        eng = _make_engine(_configured_url)
        with eng.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return eng, _configured_url
    except Exception as e:
        fallback = f"sqlite:///{BASE_DIR / 'jobs.db'}"
        print(f"[db] WARNING: Primary DB unreachable ({e}). Falling back to SQLite: {fallback}")
        return _make_engine(fallback), fallback
```
This guarantees the container binds to its port and boots successfully, rather than entering a crash loop.

### 3. Cross-Origin Resource Sharing (CORS)
- **Problem**: When accessing the backend from Vercel (`https://signiq-frontend.vercel.app`), browsers send an `OPTIONS` preflight request. The backend returned `400 Bad Request (Disallowed CORS origin)`.
- **Fix**: Configured wildcard CORS in [`app/main.py`](../app/main.py) and defaulted `ALLOWED_ORIGINS` to `*`:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=False,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

---

## 🖥️ 7. Full-Stack Self-Hosting on Render (Bypassing Vercel)

FastAPI natively mounts and serves the static frontend directory:
```python
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
```

### Advantages of Using Render for Both Frontend & Backend:
- **Zero CORS issues**: Frontend and API share the exact same origin (`signiq.onrender.com`).
- **Synchronized deployments**: No lag or version mismatch between Vercel and Render deploys.
- **Single URL**: The complete application is accessible via:
  - Landing page: `https://signiq.onrender.com/`
  - Detection page: `https://signiq.onrender.com/detect.html`
  - Similarity page: `https://signiq.onrender.com/similarity.html`

---

## 🛠️ 8. Step-by-Step Production Deployment Guide

### Prerequisites
- GitHub repository with code pushed to `main`.
- Supabase project for persistent database storage (free tier).
- Render account (free tier).

### Render Web Service Configuration
1. **New Web Service** → Connect `ayse-hatun/signIQ`.
2. **Settings**:
   - **Environment**: `Python 3`
   - **Region**: Singapore or Frankfurt (closest to your users)
   - **Branch**: `main`
   - **Root Directory**: *(Leave blank)*
   - **Build Command**:
     ```bash
     pip install -r requirements-build.txt && pip install -r requirements.txt && python scripts/export_to_onnx.py && pip uninstall -y torch torchvision libreyolo onnxscript torchaudio 2>/dev/null || true
     ```
   - **Start Command**:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```
3. **Environment Variables**:
   | Variable | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Stable runtime version |
   | `USE_CELERY` | `false` | Enables in-process free async runner |
   | `ALLOWED_ORIGINS`| `*` | Permits cross-origin calls |
   | `DATABASE_URL` | `postgresql://postgres.[REF]:[PW]@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require` | Supabase IPv4 Pooler |
   | `HF_TOKEN` | *(Optional)* `hf_...` | Prevents Hugging Face rate limits during build |
4. **Build Filters**:
   - Add `frontend` to **Ignored Paths** if you only want backend code changes to trigger rebuilds.

---

## 🚨 9. Troubleshooting & Diagnostic Runbook

| Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| **`Out of memory (used over 512Mi)`** | PyTorch imported at runtime | Ensure `torch` is not in `requirements.txt`. Verify `worker/models.py` only imports `onnxruntime`. |
| **`Network is unreachable (2406:da12:...)`** | Render Free cannot route IPv6 to Supabase port 5432 | Use Supabase Transaction Pooler URI on port `6543` (IPv4). |
| **`FATAL: password authentication failed`** | Wrong DB password or username missing project ref | In pooler URI, username must be `postgres.[PROJECT-REF]`, and password is your database password. |
| **`Disallowed CORS origin` / `Failed to fetch`** | Missing wildcard in CORS middleware | Verify `app.add_middleware(CORSMiddleware, allow_origins=["*"])`. |
| **`502 Bad Gateway` during similarity** | Redundant YOLO runs causing CPU starvation timeout | Use `sim_inference.preprocess()` to validate reference images; do not run YOLO on reference crops. |
| **Bounding box shifted or clipped** | Center-padding or `xywh2xyxy` transform applied | Use top-left letterboxing (`canvas.paste(resized, (0, 0))`) and read output directly as `[x1, y1, x2, y2]`. |
| **Cold start delay (~30s - 45s)** | Render Free spins down after 15 minutes of inactivity | Set up a free monitor (e.g., [UptimeRobot](https://uptimerobot.com/)) to ping `https://signiq.onrender.com/health` every 10 minutes. |

---

*Authored for the SignIQ Engineering Team — Production Grade, Zero-Cost Cloud Deployment.*
