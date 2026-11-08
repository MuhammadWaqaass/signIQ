import ssl

from celery import Celery

from app.config import REDIS_URL

celery_app = Celery("signature", broker=REDIS_URL, include=["worker.tasks"])

celery_app.conf.update(
    task_ignore_result=True,          # status lives in the database, not Redis (saves Redis commands)
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

if REDIS_URL.startswith("rediss://"):  # Upstash uses TLS
    celery_app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
