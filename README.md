<div align="center">

# 🖋️ SignIQ
### Automated Signature Detection & Siamese Verification Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Celery](https://img.shields.io/badge/Celery-5.4+-37814A.svg?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Upstash_Redis-Queue-red.svg?style=for-the-badge&logo=redis&logoColor=white)](https://upstash.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>SignIQ</b> is a production-grade, AI-driven signature detection and verification platform. It accurately detects signatures within complex scanned documents, isolates signature regions, and verifies authenticity against reference signatures using deep Siamese feature representations.
</p>

[Key Features](#-key-features) •
[Tech Stack](#-tech-stack) •
[Architecture](#-system-architecture) •
[Getting Started](#-getting-started) •
[API Reference](#-api-reference) •
[Contributing](#-contributing) •
[License](#-license)

---

</div>

## 🌟 Key Features

- **🔍 Precise Signature Detection (LibreYOLO9m)**
  - Detects single or multiple signatures across scanned contracts, applications, and legal forms.
  - Generates bounding boxes, coordinates, and detection confidence metrics.
  - Automatically crops and persists isolated signature candidates.

- **⚖️ Siamese Similarity Verification (ConvNeXt-Tiny)**
  - Compares detected signatures with a target reference signature in deep feature space.
  - Produces an exact percentage similarity score.
  - Employs a calibrated $\ge 78\%$ cutoff threshold for definitive classification:
    - **`✔ SIMILAR`** — High confidence match; same signer.
    - **`✖ NOT SIMILAR`** — Divergent features; distinct signer / potential forgery.

- **🛡️ Pre-Matching Signature Validation**
  - Validates document and reference inputs for signature presence and stroke density before triggering Siamese comparison.
  - Prevents erroneous comparisons when non-signature images (e.g. blank pages, photos, logos) are submitted.

- **⚡ Asynchronous Queue & Micro-Service Architecture**
  - Decoupled API and worker execution via **Celery** and **Upstash Redis**.
  - Models preloaded into worker memory on startup for sub-second inference.

- **🎨 Neo-Brutalist Responsive Frontend**
  - Clean HTML5 / CSS3 / ES6 user interface served natively by FastAPI.
  - Interactive canvas preview for bounding-box visualization on documents.
  - Side-by-side dual signature comparison cards with animated match bars.

- **🗄️ Flexible Persistence**
  - Ships with zero-config **SQLite** for rapid local development.
  - Seamlessly upgrades to cloud-native **Supabase PostgreSQL** via SQLAlchemy.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Client["Client Browser / API Consumer"]
    
    subgraph FastAPI_Backend ["FastAPI Server (Port 8000)"]
        Router["/api/v1/jobs Router"]
        Static["Static Files & Web UI"]
        Storage["Local / Cloud Storage Engine"]
    end
    
    subgraph Celery_Worker ["Async Celery Worker"]
        Queue["Upstash Redis Queue"]
        Detector["LibreYOLO9m Detector (best.pt)"]
        Similarity["ConvNeXt-Tiny Siamese Model"]
    end
    
    subgraph Database ["Persistence Layer"]
        DB[(SQLite / Supabase Postgres)]
    end

    Client -->|Upload Document & Reference| Router
    Router -->|Store Files| Storage
    Router -->|Create Job Record| DB
    Router -->|Enqueue Job ID| Queue
    Queue -->|Consume Task| Celery_Worker
    Celery_Worker -->|Detect Signatures| Detector
    Celery_Worker -->|Verify Similarity| Similarity
    Celery_Worker -->|Update Status & Results| DB
    Client -->|Poll Status & Fetch Crops| Router
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance asynchronous REST API framework |
| **Worker Queue** | [Celery](https://docs.celeryq.dev/) | Distributed asynchronous task queue |
| **Broker** | [Upstash Redis](https://upstash.com/) | Serverless managed Redis message broker |
| **Detection AI** | [LibreYOLO9m](https://github.com/ayse-hatun/signIQ) | YOLOv9-based signature localization model |
| **Similarity AI** | [ConvNeXt-Tiny](https://huggingface.co/izamscientist/signature-similarity-convnext-tiny) | Siamese neural network for stroke similarity matching |
| **ORM / DB** | [SQLAlchemy](https://www.sqlalchemy.org/) | SQLite (local) / Supabase PostgreSQL (production) |
| **Frontend** | Vanilla JS / CSS3 / HTML5 | Responsive neo-brutalist interactive UI |

---

## 📁 Project Structure

```text
signIQ/
├── app/
│   ├── celery_app.py        # Celery task queue configuration
│   ├── config.py            # Environment configuration & defaults
│   ├── db.py                # Database models & SQLAlchemy session setup
│   ├── main.py              # FastAPI application & route declarations
│   ├── schemas.py           # Pydantic validation schemas
│   └── storage.py          # File storage & asset URL helpers
├── docs/
│   ├── MODEL.md             # Deep learning architecture specifications
│   ├── PRD.md               # Product Requirements Document
│   └── SRS.md               # Software Requirements Specification
├── frontend/
│   ├── css/
│   │   └── style.css        # Neo-brutalist styling & responsive layouts
│   ├── js/
│   │   ├── api.js           # REST API client wrapper & polling logic
│   │   ├── config.js        # Dynamic API base URL configuration
│   │   ├── main.js          # Navigation & drawer behavior
│   │   └── tool.js          # Interactive canvas & result rendering
│   ├── detect.html          # Document detection view
│   ├── index.html           # Landing page & feature breakdown
│   └── similarity.html      # Side-by-side signature comparison view
├── weights/
│   └── best.pt              # LibreYOLO model weights (downloaded on setup)
├── worker/
│   ├── models.py            # Model loaders & cache management
│   ├── pipeline.py          # Pure computer-vision inference functions
│   └── tasks.py             # Celery background job orchestration
├── .env.example             # Template environment variables
├── .gitignore               # Strict exclusion of secrets, weights, and temp data
├── requirements.txt         # Pinned Python package dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10 – 3.13** installed.
- An **Upstash Redis** database (free tier available at [upstash.com](https://upstash.com/)) or a local Redis instance.
- Git.

### 1. Clone the Repository
```bash
git clone https://github.com/ayse-hatun/signIQ.git
cd signIQ
```

### 2. Set Up Virtual Environment & Dependencies

On **Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

On **Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create your local `.env` from the provided template:
```bash
cp .env.example .env     # On Linux / macOS
copy .env.example .env   # On Windows
```

Edit `.env` with your settings:
```ini
# Upstash Redis connection string
REDIS_URL=rediss://default:YOUR_PASSWORD@YOUR_HOST.upstash.io:6379

# Local data & database
DATA_DIR=./data
DATABASE_URL=sqlite:///./jobs.db

# Model configuration
DETECTOR_LOCAL_PATH=./weights/best.pt
DETECTOR_REPO=ayse67895/Signature-detection
DETECTOR_FILENAME=best.pt
SIMILARITY_REPO=izamscientist/signature-similarity-convnext-tiny

# Thresholds
DETECT_CONF=0.385
DETECT_IOU=0.5
```

> [!TIP]
> If `weights/best.pt` is not present locally, SignIQ automatically pulls it from Hugging Face Hub during first worker boot.

---

## 💻 Running the Application

SignIQ requires two processes running concurrently:

### Terminal 1: FastAPI Web & API Server
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Launch server
uvicorn app.main:app --port 8000 --reload
```
The application will be live at:
- **Web App**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Signature Detection Page**: [http://127.0.0.1:8000/detect.html](http://127.0.0.1:8000/detect.html)
- **Signature Similarity Page**: [http://127.0.0.1:8000/similarity.html](http://127.0.0.1:8000/similarity.html)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Terminal 2: Celery Inference Worker

On **Windows** (uses `--pool=solo`):
```powershell
.venv\Scripts\activate
celery -A app.celery_app worker --pool=solo --loglevel=info
```

On **Linux / macOS**:
```bash
source .venv/bin/activate
celery -A app.celery_app worker --loglevel=info
```

---

## 📡 API Reference

### Health Check
```http
GET /health
```
**Response**: `{"status": "ok"}`

---

### Submit Verification / Detection Job
```http
POST /api/v1/jobs
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `document` | File (PNG/JPEG/WebP) | **Yes** | Scanned document or isolated signature image |
| `reference` | File (PNG/JPEG/WebP) | **No** | Target signature to compare against (similarity mode) |

**Sample Response (`202 Accepted`)**:
```json
{
  "job_id": "1fa26097-9f53-41b0-b5a2-acadb2223e64",
  "status": "queued"
}
```

---

### Query Job Status & Results
```http
GET /api/v1/jobs/{job_id}
```

**Completed Response (`200 OK`)**:
```json
{
  "job_id": "1fa26097-9f53-41b0-b5a2-acadb2223e64",
  "status": "completed",
  "result": {
    "image_size": [1470, 980],
    "signatures_found": 1,
    "detections": [
      {
        "box": [45.0, 110.2, 380.5, 290.0],
        "confidence": 0.965,
        "label": "signature",
        "crop_url": "/files/1fa26097-9f53-41b0-b5a2-acadb2223e64/crop_1.png",
        "similarity_score": 0.8845,
        "verdict": "SIMILAR"
      }
    ],
    "reference_is_signature": true,
    "error_message": null
  },
  "error": null,
  "created_at": "2026-11-08T15:31:32.983282Z",
  "updated_at": "2026-11-08T15:31:41.451194Z"
}
```

---

## 🤝 Contributing

Contributions make the open-source community an inspiring place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

### How to Contribute

1. **Fork the Project**
   Click the **Fork** button at the top right of the repository.

2. **Clone your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/signIQ.git
   cd signIQ
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

4. **Commit your Changes**
   Follow clear, descriptive commit messages:
   ```bash
   git commit -m "feat: add batch verification endpoint"
   ```

5. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**
   Go to the original repository on GitHub and click **New Pull Request** to explain your enhancements.

### Guidelines & Rules
- **Never commit `.env` or secrets**: Keep credentials in local configuration only.
- **Do not commit large binary weights (`.pt`, `.bin`)**: Use Hugging Face Hub references.
- **Code Style**: Maintain clean PEP 8 compliance for Python code and semantic HTML/CSS formatting.

---

## 🛡️ License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <sub>Developed by <b>Ayesha Rasheed</b> and the SignIQ community.</sub>
</div>
