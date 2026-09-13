"""API tests.

The agent is stubbed out here on purpose -- what is under test is the HTTP
contract around it: upload validation, the async job shape (202 + poll), and the
promise that an uploaded dataset does not linger on disk after the run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ground_truth import api as api_module
from ground_truth.schemas import (
    AuditReport,
    AuditSummary,
    Category,
    Finding,
    Severity,
)

CSV_BYTES = b"customer_id,age\nC1,23\nC2,999\n"


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_module.app)


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test gets an empty run registry."""
    api_module._RUNS.clear()
    api_module._PROGRESS.clear()
    yield
    api_module._RUNS.clear()
    api_module._PROGRESS.clear()


class _FakeAgent:
    """Stands in for AuditAgent: returns a finished report without a model."""

    def __init__(self, *_args, **_kwargs):
        pass

    def audit(self, data_path: Path, user_context: str = "", on_progress=None) -> AuditReport:
        if on_progress:
            on_progress("Profiling dataset...")
        assert data_path.exists(), "the agent must be handed a file that is still there"
        return AuditReport(
            run_id="replaced-by-caller",
            dataset_name=data_path.name,
            status="completed",
            findings=[
                Finding(
                    title="Age contains placeholder 999",
                    category=Category.MISSING_DATA,
                    severity=Severity.HIGH,
                    columns=["age"],
                    evidence="1 of 2 rows (50%) have age == 999",
                    why_it_matters="999 will be treated as a real age.",
                    recommendation="Replace with NaN.",
                    fix_code="df.loc[df['age'] == 999, 'age'] = pd.NA",
                )
            ],
            summary=AuditSummary(
                overall_risk=Severity.HIGH,
                summary="Placeholder values present.",
                ready_for_modeling=False,
                next_steps=["Clean age"],
            ),
        )


@pytest.fixture
def stub_agent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_module, "AuditAgent", _FakeAgent)


# ---------- health ----------

def test_health_reports_the_model_that_will_actually_run(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    settings = api_module.settings
    expected = settings.ollama_model if settings.provider == "ollama" else settings.model
    assert body["model"] == expected
    assert body["provider"] == settings.provider


# ---------- upload validation ----------

def test_unsupported_extension_is_rejected(client: TestClient):
    r = client.post("/audits", files={"file": ("notes.json", b"{}", "application/json")})
    assert r.status_code == 400
    assert ".json" in r.json()["detail"]


def test_empty_upload_is_rejected(client: TestClient):
    r = client.post("/audits", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_oversized_upload_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """The cap exists so one upload cannot exhaust memory or disk."""
    monkeypatch.setattr(api_module.settings, "max_upload_mb", 0)
    r = client.post("/audits", files={"file": ("big.csv", CSV_BYTES, "text/csv")})
    assert r.status_code == 413


def test_oversized_upload_leaves_nothing_behind(client: TestClient, monkeypatch):
    before = set(Path(tempfile.gettempdir()).glob("dd_*"))
    monkeypatch.setattr(api_module.settings, "max_upload_mb", 0)
    client.post("/audits", files={"file": ("big.csv", CSV_BYTES, "text/csv")})
    assert set(Path(tempfile.gettempdir()).glob("dd_*")) == before


# ---------- the async job shape ----------

def test_audit_runs_and_is_retrievable(client: TestClient, stub_agent):
    r = client.post(
        "/audits",
        files={"file": ("messy.csv", CSV_BYTES, "text/csv")},
        data={"context": "churn data"},
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    assert r.json()["status"] == "running"

    # TestClient runs background tasks before returning, so the audit is done.
    status = client.get(f"/audits/{run_id}").json()
    assert status["status"] == "completed"
    assert status["report"]["run_id"] == run_id, "the caller's id must win"
    assert len(status["report"]["findings"]) == 1


def test_uploaded_data_is_deleted_after_the_run(client: TestClient, stub_agent):
    """User data must not sit in temp after the audit finishes."""
    run_id = client.post(
        "/audits", files={"file": ("messy.csv", CSV_BYTES, "text/csv")}
    ).json()["run_id"]
    assert not list(Path(tempfile.gettempdir()).glob(f"dd_{run_id}*"))


def test_original_filename_is_never_used_as_a_path(client: TestClient, stub_agent):
    """Path traversal via the upload name: the name is reported, never joined."""
    run_id = client.post(
        "/audits", files={"file": ("../../evil.csv", CSV_BYTES, "text/csv")}
    ).json()["run_id"]
    report = client.get(f"/audits/{run_id}").json()["report"]
    assert report["dataset_name"] == "evil.csv"

    # ...and the generated script must load the user's file, not the temp copy.
    script = client.get(f"/audits/{run_id}/fix_script.py").text
    assert "evil.csv" in script
    assert "dd_" not in script


def test_a_failing_agent_surfaces_as_a_failed_report(client: TestClient, monkeypatch):
    class _Boom(_FakeAgent):
        def audit(self, *_a, **_k):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(api_module, "AuditAgent", _Boom)
    run_id = client.post(
        "/audits", files={"file": ("messy.csv", CSV_BYTES, "text/csv")}
    ).json()["run_id"]

    body = client.get(f"/audits/{run_id}").json()
    assert body["status"] == "failed"
    assert "model exploded" in body["report"]["error"]


def test_unknown_run_id_is_404(client: TestClient):
    assert client.get("/audits/deadbeef").status_code == 404
    assert client.get("/audits/deadbeef/report.md").status_code == 404
    assert client.get("/audits/deadbeef/fix_script.py").status_code == 404


def test_artifacts_are_409_while_the_audit_is_running(client: TestClient):
    api_module._RUNS["r1"] = AuditReport(run_id="r1", dataset_name="x.csv", status="running")
    assert client.get("/audits/r1/report.md").status_code == 409
    assert client.get("/audits/r1/fix_script.py").status_code == 409


def test_report_and_fix_script_render(client: TestClient, stub_agent):
    run_id = client.post(
        "/audits", files={"file": ("messy.csv", CSV_BYTES, "text/csv")}
    ).json()["run_id"]

    md = client.get(f"/audits/{run_id}/report.md").text
    assert "# Data Audit" in md
    assert "Age contains placeholder 999" in md

    script = client.get(f"/audits/{run_id}/fix_script.py").text
    assert "import pandas as pd" in script
    assert "df.loc[df['age'] == 999" in script
