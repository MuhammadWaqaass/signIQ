from datetime import datetime, timezone
import os

from sqlalchemy import JSON, DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import DATABASE_URL as _configured_url

# ---------------------------------------------------------------------------
# Engine — with IPv6 fallback to SQLite
# ---------------------------------------------------------------------------

def _make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def _resolve_engine():
    """Try the configured DATABASE_URL; if it's unreachable at startup fall back
    to a local SQLite file so the app can still boot and serve health checks."""
    if _configured_url.startswith("sqlite"):
        return _make_engine(_configured_url), _configured_url

    try:
        eng = _make_engine(_configured_url)
        # Quick connectivity probe
        with eng.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        print(f"[db] Connected to database: {_configured_url[:40]}...")
        return eng, _configured_url
    except Exception as e:
        fallback = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'jobs.db')}"
        print(f"[db] WARNING: Primary DB unreachable ({e}). Falling back to SQLite: {fallback}")
        return _make_engine(fallback), fallback


engine, _active_url = _resolve_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|processing|completed|failed
    document_key: Mapped[str] = mapped_column(String(300))
    reference_key: Mapped[str | None] = mapped_column(String(300), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


def init_db():
    try:
        Base.metadata.create_all(engine)
        print("[db] Tables created/verified OK")
    except Exception as e:
        print(f"[db] WARNING: Could not create tables: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
