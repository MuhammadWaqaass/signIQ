import uuid

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app import storage
from app.celery_app import celery_app
from app.config import ALLOWED_ORIGINS, ALLOWED_TYPES, BASE_DIR, DATA_DIR, MAX_UPLOAD_MB
from app.db import Job, get_db, init_db
from app.schemas import JobCreated, JobOut

DATA_DIR.mkdir(parents=True, exist_ok=True)
init_db()

app = FastAPI(title="Signature Detection + Similarity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves saved crops so the frontend can display them (local dev only).
app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")


def read_image(upload: UploadFile) -> tuple[bytes, str]:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"{upload.filename}: only PNG, JPEG or WebP images are allowed")
    data = upload.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{upload.filename}: larger than {MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, f"{upload.filename}: empty file")
    return data, ALLOWED_TYPES[upload.content_type]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/jobs", response_model=JobCreated, status_code=202)
def create_job(
    document: UploadFile = File(..., description="Document image to scan for signatures"),
    reference: UploadFile | None = File(None, description="Reference signature to compare against (optional)"),
    db: Session = Depends(get_db),
):
    doc_bytes, doc_ext = read_image(document)

    job_id = str(uuid.uuid4())
    doc_key = f"{job_id}/document{doc_ext}"
    storage.save_bytes(doc_key, doc_bytes)

    ref_key = None
    if reference and reference.filename:
        ref_bytes, ref_ext = read_image(reference)
        ref_key = f"{job_id}/reference{ref_ext}"
        storage.save_bytes(ref_key, ref_bytes)

    job = Job(id=job_id, status="queued", document_key=doc_key, reference_key=ref_key)
    db.add(job)
    db.commit()

    try:
        # Only the job id goes through Redis. The worker reads everything else from the database.
        celery_app.send_task("tasks.process_job", args=[job_id])
    except Exception as e:
        job.status = "failed"
        job.error = f"Could not enqueue job: {e}"
        db.commit()
        raise HTTPException(503, "Queue unavailable, try again shortly")

    return JobCreated(job_id=job_id, status="queued")


@app.get("/api/v1/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobOut(
        job_id=job.id,
        status=job.status,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

