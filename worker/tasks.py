import traceback

from celery.signals import worker_ready
from PIL import Image

from app import config, storage
from app.celery_app import celery_app
from app.db import Job, SessionLocal
from worker import models, pipeline


@worker_ready.connect
def preload_models(**kwargs):
    # Load both models when the worker boots, so the first job is not slow.
    models.load_all()


@celery_app.task(name="tasks.process_job")
def process_job(job_id: str):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "processing"
        db.commit()

        doc_path = storage.path_for(job.document_key)
        ref_path = storage.path_for(job.reference_key) if job.reference_key else None

        detector = models.get_detector()
        sim_model, sim_inference = models.get_similarity()

        # Detect signatures on document
        detections = pipeline.detect(detector, doc_path, config.DETECT_CONF, config.DETECT_IOU)
        if len(detections) == 0 and ref_path:
            # Fallback for isolated signature crops uploaded as document
            low_dets = pipeline.detect(detector, doc_path, 0.20, config.DETECT_IOU)
            if len(low_dets) > 0:
                detections = low_dets

        document = Image.open(doc_path).convert("RGB")

        # In similarity mode, verify if reference image is a signature before matching
        ref_is_signature = True
        ref_error = None
        if ref_path:
            ref_dets = pipeline.detect(detector, ref_path, conf=0.15, iou=config.DETECT_IOU)
            if len(ref_dets) == 0:
                try:
                    sim_inference.preprocess(str(ref_path))
                    ref_dets_low = pipeline.detect(detector, ref_path, conf=0.05, iou=config.DETECT_IOU)
                    if len(ref_dets_low) == 0:
                        ref_is_signature = False
                        ref_error = "Reference image is not recognized as a signature. Please upload a clear signature image."
                except Exception:
                    ref_is_signature = False
                    ref_error = "Reference image is blank or not a valid signature image."

        for i, det in enumerate(detections, start=1):
            crop_key = f"{job_id}/crop_{i}.png"
            crop_path = pipeline.crop_and_save(document, det["box"], storage.path_for(crop_key))
            det["crop_url"] = storage.url_for(crop_key)
            if ref_path:
                if ref_is_signature:
                    score, verdict = pipeline.compare(sim_model, sim_inference, crop_path, ref_path)
                    det["similarity_score"] = score
                    det["verdict"] = verdict  # Strictly SIMILAR or NOT_SIMILAR
                else:
                    det["similarity_score"] = None
                    det["verdict"] = "INVALID_REFERENCE"

        job.result = {
            "image_size": list(document.size),  # [width, height], for drawing boxes
            "signatures_found": len(detections),
            "detections": detections,
            "reference_is_signature": ref_is_signature if ref_path else None,
            "error_message": ref_error if (ref_path and not ref_is_signature) else (
                "No signature detected on the document image." if len(detections) == 0 else None
            ),
        }
        job.status = "completed"
        db.commit()
    except Exception as e:
        db.rollback()
        job = db.get(Job, job_id)
        if job:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            db.commit()
        traceback.print_exc()
    finally:
        db.close()
