"""File storage. Local folder for now.

Everything goes through these functions, so moving to Supabase/R2 later means
changing only this file. Note: when API and worker run on DIFFERENT machines
(deployment), they can no longer share a local folder, so switch this to
Supabase/R2 before deploying.
"""
from pathlib import Path

from app.config import DATA_DIR


def path_for(key: str) -> Path:
    p = (DATA_DIR / key).resolve()
    if DATA_DIR not in p.parents:  # block path tricks like ../../
        raise ValueError("bad storage key")
    return p


def save_bytes(key: str, data: bytes) -> Path:
    p = path_for(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def url_for(key: str) -> str:
    return f"/files/{key}"
