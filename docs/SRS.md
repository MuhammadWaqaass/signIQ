# Software Requirements Specification (SRS)
**System:** Signature Detection and Verification Service
**Version:** 1.0 (draft)

---

## 1. Introduction

### 1.1 Purpose
This document specifies what the backend must do and how it behaves. It supports the PRD (`PRD.md`) and the model descriptions in `MODEL.md`.

### 1.2 Definitions
| Term | Meaning |
|---|---|
| Job | One request to analyse a document against a reference signature |
| Detector | Object-detection model that finds signatures on a page |
| Similarity model | Siamese model that scores how alike two signatures are (0 to 1) |
| Worker | Background process that runs the models |
| Queue | Redis, holds job IDs waiting for a worker |

### 1.3 Implementation status
Each requirement is marked **[Done]** (in the current code), **[Local]** (works only on one machine) or **[Planned]**.

## 2. Overall description

### 2.1 Architecture
```
Frontend ──► FastAPI ──► Storage (files)
                │   └──► Database (job row, status)
                └──► Redis queue (job_id only)
                           │
                       Worker ──► Detector + Similarity model (from HF Hub)
                           ├──► Storage (crops)
                           └──► Database (result)
Frontend polls GET /api/v1/jobs/{id}
```

### 2.2 Components
| Component | Technology | Status |
|---|---|---|
| API | FastAPI, Uvicorn | [Done] |
| Queue | Celery with Redis (Upstash) | [Done] |
| Worker | Celery worker, `--pool=solo` on Windows | [Done] |
| Database | SQLAlchemy, SQLite now, Postgres later | [Done] SQLite / [Planned] Postgres |
| File storage | Local folder now, Supabase or R2 later | [Local] |
| Model registry | Hugging Face Hub | [Done] similarity / [Planned] detector |
| Frontend | React or Next.js | [Planned] |
| Training pipeline | HF dataset trigger, remote training, Comet or MLflow | [Planned] |

### 2.3 Assumptions
- Input documents are clear images (scans or flat photos).
- One worker process handles one job at a time (`concurrency=1`).
- The reference image contains a single signature.

## 3. Functional requirements

### 3.1 Job submission
| ID | Requirement | Status |
|---|---|---|
| FR-1 | The API shall accept a `document` image and a `reference` image as multipart form data. | [Done] |
| FR-2 | Only PNG, JPEG and WebP shall be accepted; others get HTTP 415. | [Done] |
| FR-3 | Files larger than 10 MB shall be rejected with HTTP 413; empty files with HTTP 400. | [Done] |
| FR-4 | The API shall store both files, create a job record with status `queued`, enqueue the job ID, and return HTTP 202 with `job_id`. | [Done] |
| FR-5 | The API shall not run any model. | [Done] |
| FR-6 | If the queue is unreachable the job shall be marked `failed` and the API shall return HTTP 503. | [Done] |

### 3.2 Job status
| ID | Requirement | Status |
|---|---|---|
| FR-7 | `GET /api/v1/jobs/{job_id}` shall return status, result (if completed), error (if failed) and timestamps. | [Done] |
| FR-8 | An unknown job ID shall return HTTP 404. | [Done] |
| FR-9 | Status values shall be `queued`, `processing`, `completed`, `failed`. | [Done] |

### 3.3 Processing
| ID | Requirement | Status |
|---|---|---|
| FR-10 | The worker shall load both models once at startup, not per job. | [Done] |
| FR-11 | The worker shall detect signatures using confidence 0.385 and IoU 0.5 (configurable). | [Done] |
| FR-12 | For each detection, the worker shall crop the box, save the crop, and compute a similarity score against the reference. | [Done] |
| FR-13 | The worker shall attach the verdict ACCEPT / REVIEW / REJECT from the similarity model's thresholds. | [Done] |
| FR-14 | The worker shall write the result and set status `completed`. | [Done] |
| FR-15 | On any error the worker shall set status `failed` with a message and keep running. | [Done] |
| FR-16 | A document with no signatures shall complete with an empty detection list. | [Done] |

### 3.4 Health
| ID | Requirement | Status |
|---|---|---|
| FR-17 | `GET /health` shall return `{"status": "ok"}` without touching the queue or models. | [Done] |

### 3.5 Frontend
| ID | Requirement | Status |
|---|---|---|
| FR-18 | The UI shall upload both images and show job progress by polling. | [Planned] |
| FR-19 | The UI shall draw boxes on the document with score and verdict labels. | [Planned] |
| FR-20 | The UI shall show cropped signatures next to the reference. | [Planned] |

### 3.6 Training pipeline
| ID | Requirement | Status |
|---|---|---|
| FR-21 | Uploading new data to the Hugging Face dataset shall trigger a remote training job. | [Planned] |
| FR-22 | Each run shall log parameters and metrics to Comet or MLflow. | [Planned] |
| FR-23 | The new model shall be pushed to the Hub and deployed only if its metrics beat the current model. | [Planned] |
| FR-24 | The worker shall be able to pick up a new model version without code changes. | [Planned] |

## 4. External interfaces

### 4.1 POST /api/v1/jobs
Request: `multipart/form-data` with fields `document` (image) and `reference` (image).

Response `202`:
```json
{ "job_id": "3f2b...", "status": "queued" }
```

### 4.2 GET /api/v1/jobs/{job_id}
Response `200` (completed):
```json
{
  "job_id": "3f2b...",
  "status": "completed",
  "result": {
    "image_size": [1654, 2339],
    "signatures_found": 1,
    "detections": [
      {
        "box": [412.0, 1880.5, 780.2, 2010.9],
        "confidence": 0.91,
        "label": "signature",
        "crop_url": "/files/3f2b.../crop_1.png",
        "similarity_score": 0.84,
        "verdict": "REVIEW"
      }
    ]
  },
  "error": null,
  "created_at": "...",
  "updated_at": "..."
}
```
Box format is `[x1, y1, x2, y2]` in pixels of the original document. Values above are illustrative.

### 4.3 Error codes
| Code | Cause |
|---|---|
| 400 | Empty file |
| 404 | Job not found |
| 413 | File over 10 MB |
| 415 | Unsupported image type |
| 422 | Missing form field |
| 503 | Queue unavailable |

## 5. Data model
Table `jobs`:
| Column | Type | Notes |
|---|---|---|
| id | string(36) | UUID, primary key |
| status | string | queued / processing / completed / failed |
| document_key | string | Storage key of the document |
| reference_key | string | Storage key of the reference |
| result | JSON | Detections, nullable |
| error | text | Failure message, nullable |
| created_at, updated_at | timestamp | UTC |

The database is the source of truth for status. Redis carries only the job ID.

## 6. Non-functional requirements
| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | The API shall respond to job submission within 1 second under normal load. |
| NFR-2 | Performance | End-to-end processing time for one page image shall be measured and reported (CPU worker). |
| NFR-3 | Scalability | Throughput shall scale by adding workers; the API stays unchanged. |
| NFR-4 | Reliability | A failed job shall never stop the worker; errors are stored on the job. |
| NFR-5 | Security | Uploads shall be type- and size-checked. Storage keys shall be validated against path traversal. |
| NFR-6 | Security | CORS shall be limited to the frontend origin in production (`ALLOWED_ORIGINS`). |
| NFR-7 | Security | Secrets (Redis URL, DB URL, HF token) shall come from environment variables, never from source control. |
| NFR-8 | Privacy | Signatures are personal data. Uploaded files and crops shall have a retention policy before any public launch. [Planned] |
| NFR-9 | Portability | The backend shall run on Windows, macOS and Linux. |
| NFR-10 | Licensing | All production libraries shall be MIT, Apache or similarly permissive. No Ultralytics in production. |
| NFR-11 | Observability | The worker shall log model loading, job start and failures. |
| NFR-12 | Testability | Model logic shall be in `worker/pipeline.py`, free of queue and web code. |

## 7. Constraints and known limitations
- Local file storage works only when the API and worker share one machine. Deployment needs Supabase or R2.
- Free-tier Redis has a daily command limit; Celery polls even when idle.
- Free hosts may sleep when idle; the first request after sleep is slow.
- The similarity model was trained on scans; phone photos may score lower.
- No authentication or rate limiting in v1.

## 8. Traceability
| PRD item | SRS requirements |
|---|---|
| P1 Upload | FR-1 to FR-3 |
| P2 Detect | FR-11, FR-16 |
| P3 Similarity | FR-12, FR-13 |
| P4 Async jobs | FR-4 to FR-9, FR-14, FR-15 |
| P5 Frontend | FR-18 to FR-20 |
| P7 Training pipeline | FR-21 to FR-24 |
