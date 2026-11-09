import gc
import traceback

from celery.signals import worker_ready
from PIL import Image

from app import config, storage
from app.celery_app import celery_app
from app.db import Job, SessionLocal
from worker import models, pipeline


@worker_ready.connect
def preload_models(**kwargs):
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
        sim_model, sim_inference = models.get_similarity() if ref_path else (None, None)

        # Detect signatures on document
        detections = pipeline.detect(detector, doc_path, config.DETECT_CONF, config.DETECT_IOU)
        if len(detections) == 0 and ref_path:
            # Fallback for isolated signature crops uploaded as document
            low_dets = pipeline.detect(detector, doc_path, 0.20, config.DETECT_IOU)
            if len(low_dets) > 0:
                detections = low_dets

        # If document is an isolated signature crop and YOLO didn't locate a sub-box,
        # treat the entire document image as the signature crop for comparison
        document = Image.open(doc_path).convert("RGB")
        w, h = document.size
        if len(detections) == 0 and ref_path:
            try:
                sim_inference.preprocess(str(doc_path))
                detections = [{"box": [0.0, 0.0, float(w), float(h)], "confidence": 1.0, "label": "signature"}]
            except Exception:
                pass

        # Validate reference image
        ref_is_signature = True
        ref_error = None
        if ref_path:
            try:
                sim_inference.preprocess(str(ref_path))
            except Exception:
                ref_is_signature = False
                ref_error = "Reference image is blank or not a valid signature image."

        # Free temporary buffer memory before running similarity model
        gc.collect()

        for i, det in enumerate(detections, start=1):
            crop_key = f"{job_id}/crop_{i}.png"
            crop_path = pipeline.crop_and_save(document, det["box"], storage.path_for(crop_key))
            det["crop_url"] = storage.url_for(crop_key)
            if ref_path:
                if ref_is_signature:
                    try:
                        score, verdict = pipeline.compare(sim_model, sim_inference, crop_path, ref_path)
                        det["similarity_score"] = score
                        det["verdict"] = verdict
                    except Exception as cmp_err:
                        print(f"Similarity comparison failed for crop {i}: {cmp_err}")
                        det["similarity_score"] = None
                        det["verdict"] = "ERROR"
                else:
                    det["similarity_score"] = None
                    det["verdict"] = "INVALID_REFERENCE"

        job.result = {
            "image_size": [w, h],
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
