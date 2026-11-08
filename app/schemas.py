from datetime import datetime

from pydantic import BaseModel


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobOut(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
