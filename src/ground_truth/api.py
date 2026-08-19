"""FastAPI backend.

WHY BACKGROUND JOBS: an audit takes 30-90 seconds. A synchronous endpoint would
hit proxy/browser timeouts and block a worker. So: POST returns a run_id
immediately, the client polls GET. This is the standard shape for any
LLM-agent-backed API and is worth being able to explain.

WHY A DICT + JSON FILES FOR STORAGE: this is a single-node demo. Swapping in
Redis + Postgres is a repository change, not a redesign. Over-engineering it
here would add infrastructure to the README for zero demonstrated skill.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from threading import Lock

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .agent import AuditAgent
from .config import get_settings
from .logging_setup import get_logger, setup_logging
from .schemas import AuditReport

settings = get_settings()
setup_logging(settings.log_level, settings.log_json)
log = get_logger(__name__)

app = FastAPI(
    title="Ground Truth",
    description="An autonomous agent that audits datasets for quality problems.",
    version="0.1.0",
)

_RUNS: dict[str, AuditReport] = {}
_PROGRESS: dict[str, str] = {}
_LOCK = Lock()

ALLOWED_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls", ".txt"}


class StartResponse(BaseModel):
    run_id: str
    status: str


class StatusResponse(BaseModel):
    run_id: str
    status: str
    progress: str
    report: AuditReport | None = None


def _persist(report: AuditReport) -> None:
    path = settings.storage_dir / f"{report.run_id}.json"
    path.write_text(report.model_dump_json(indent=2))


def _run_audit(run_id: str, tmp_path: Path, context: str) -> None:
    """Executed in a background thread by FastAPI."""
    def progress(msg: str) -> None:
        with _LOCK:
            _PROGRESS[run_id] = msg

    try:
        agent = AuditAgent(settings)
        report = agent.audit(tmp_path, user_context=context, on_progress=progress)
        report.run_id = run_id
    except Exception as e:  # never let a thread die silently
        log.exception("api.audit_failed", run_id=run_id)
        report = _RUNS[run_id]
        report.status = "failed"
        report.error = f"{type(e).__name__}: {e}"
    finally:
        tmp_path.unlink(missing_ok=True)  # don't leave user data on disk

    with _LOCK:
        _RUNS[run_id] = report
        _PROGRESS[run_id] = "done"
    _persist(report)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model}


@app.post("/audits", response_model=StartResponse, status_code=202)
async def start_audit(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    context: str = Form(default=""),
) -> StartResponse:
    """Upload a dataset and kick off an audit."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}")

    # Stream to disk with a hard size cap so a huge upload can't exhaust memory.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    run_id = uuid.uuid4().hex[:12]
    # NOTE: use the *original suffix* but never the original filename — user-
    # supplied names are a path-traversal vector.
    tmp = Path(tempfile.gettempdir()) / f"dd_{run_id}{suffix}"

    written = 0
    with tmp.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
            out.write(chunk)

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")

    safe_name = Path(file.filename or "dataset").name
    with _LOCK:
        _RUNS[run_id] = AuditReport(run_id=run_id, dataset_name=safe_name, status="running")
        _PROGRESS[run_id] = "queued"

    background.add_task(_run_audit, run_id, tmp, context[:2000])
    log.info("api.audit_started", run_id=run_id, dataset=safe_name, bytes=written)
    return StartResponse(run_id=run_id, status="running")


@app.get("/audits/{run_id}", response_model=StatusResponse)
def get_audit(run_id: str) -> StatusResponse:
    with _LOCK:
        report = _RUNS.get(run_id)
        progress = _PROGRESS.get(run_id, "")
    if report is None:
        raise HTTPException(404, "Unknown run_id.")
    return StatusResponse(
        run_id=run_id,
        status=report.status,
        progress=progress,
        report=report if report.status != "running" else None,
    )


@app.get("/audits/{run_id}/report.md", response_class=PlainTextResponse)
def get_markdown(run_id: str) -> str:
    with _LOCK:
        report = _RUNS.get(run_id)
    if report is None:
        raise HTTPException(404, "Unknown run_id.")
    if report.status == "running":
        raise HTTPException(409, "Audit still running.")
    return report.to_markdown()


@app.get("/audits/{run_id}/fix_script.py", response_class=PlainTextResponse)
def get_fix_script(run_id: str) -> str:
    with _LOCK:
        report = _RUNS.get(run_id)
    if report is None:
        raise HTTPException(404, "Unknown run_id.")
    if report.status == "running":
        raise HTTPException(409, "Audit still running.")
    return report.to_fix_script()
