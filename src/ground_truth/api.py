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

import re
import tempfile
import uuid
from collections import OrderedDict
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

# Completed reports are written to disk, so memory only has to be a cache. It is
# bounded: a long-lived server that audited thousands of files used to hold every
# report forever, which is a slow leak rather than a crash and therefore the kind
# you find in production.
MAX_RUNS_IN_MEMORY = 200

_RUNS: "OrderedDict[str, AuditReport]" = OrderedDict()
_PROGRESS: "OrderedDict[str, str]" = OrderedDict()
_LOCK = Lock()

ALLOWED_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls", ".txt"}
# run_ids are uuid4().hex[:12]. Anything else never becomes a path: this value
# arrives from the URL, and it is about to be used as a filename.
_RUN_ID = re.compile(r"^[0-9a-f]{6,32}$")


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


def _remember(run_id: str, report: AuditReport) -> None:
    """Cache a report, evicting the oldest. Caller holds _LOCK."""
    _RUNS[run_id] = report
    _RUNS.move_to_end(run_id)
    while len(_RUNS) > MAX_RUNS_IN_MEMORY:
        evicted, _ = _RUNS.popitem(last=False)
        _PROGRESS.pop(evicted, None)


def _get_report(run_id: str) -> AuditReport | None:
    """Memory first, then the JSON on disk.

    The disk read is what makes eviction invisible to a client that is still
    polling, and it means a completed run survives a restart of the API.
    """
    with _LOCK:
        report = _RUNS.get(run_id)
        if report is not None:
            _RUNS.move_to_end(run_id)
            return report

    if not _RUN_ID.match(run_id):
        return None

    path = settings.storage_dir / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return AuditReport.model_validate_json(path.read_text())
    except Exception:  # a truncated or hand-edited file is a 404, not a 500
        log.warning("api.unreadable_report", run_id=run_id)
        return None


def _run_audit(run_id: str, tmp_path: Path, context: str, dataset_name: str) -> None:
    """Executed in a background thread by FastAPI."""
    def progress(msg: str) -> None:
        with _LOCK:
            _PROGRESS[run_id] = msg

    try:
        agent = AuditAgent(settings)
        report = agent.audit(tmp_path, user_context=context, on_progress=progress)
        report.run_id = run_id
        # The agent names the report after the file it was handed, which here is
        # the temp copy. Put the user's filename back: it is what the report
        # header shows and what the generated fix script calls read_csv on.
        report.dataset_name = dataset_name
    except Exception as e:  # never let a thread die silently
        log.exception("api.audit_failed", run_id=run_id)
        report = _RUNS[run_id]
        report.status = "failed"
        report.error = f"{type(e).__name__}: {e}"
    finally:
        tmp_path.unlink(missing_ok=True)  # don't leave user data on disk

    with _LOCK:
        _remember(run_id, report)
        _PROGRESS[run_id] = "done"
    _persist(report)


@app.get("/health")
def health() -> dict[str, str]:
    # Report the model that will actually run. `settings.model` is the Anthropic
    # field, so returning it unconditionally told every local user they were
    # talking to Claude while inference happened on their own machine.
    model = settings.ollama_model if settings.provider == "ollama" else settings.model
    return {"status": "ok", "provider": settings.provider, "model": model}


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
        _remember(run_id, AuditReport(run_id=run_id, dataset_name=safe_name, status="running"))
        _PROGRESS[run_id] = "queued"

    background.add_task(_run_audit, run_id, tmp, context[:2000], safe_name)
    log.info("api.audit_started", run_id=run_id, dataset=safe_name, bytes=written)
    return StartResponse(run_id=run_id, status="running")


@app.get("/audits/{run_id}", response_model=StatusResponse)
def get_audit(run_id: str) -> StatusResponse:
    report = _get_report(run_id)
    with _LOCK:
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
    report = _get_report(run_id)
    if report is None:
        raise HTTPException(404, "Unknown run_id.")
    if report.status == "running":
        raise HTTPException(409, "Audit still running.")
    return report.to_markdown()


@app.get("/audits/{run_id}/fix_script.py", response_class=PlainTextResponse)
def get_fix_script(run_id: str) -> str:
    report = _get_report(run_id)
    if report is None:
        raise HTTPException(404, "Unknown run_id.")
    if report.status == "running":
        raise HTTPException(409, "Audit still running.")
    return report.to_fix_script()
