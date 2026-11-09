import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'jobs.db').as_posix()}")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

DETECTOR_LOCAL_PATH = os.getenv("DETECTOR_LOCAL_PATH", "")
DETECTOR_REPO = os.getenv("DETECTOR_REPO", "")
DETECTOR_FILENAME = os.getenv("DETECTOR_FILENAME", "best.pt")
SIMILARITY_REPO = os.getenv("SIMILARITY_REPO", "izamscientist/signature-similarity-convnext-tiny")

DETECT_CONF = float(os.getenv("DETECT_CONF", "0.385"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.5"))

MAX_UPLOAD_MB = 10
ALLOWED_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
