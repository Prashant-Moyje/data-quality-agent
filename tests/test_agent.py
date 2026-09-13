"""Tool, memory and agent-loop tests.

The agent test uses a FAKE Anthropic client. This is the important pattern: you
cannot write reliable tests against a non-deterministic model, so you test the
*orchestration* — does the loop dispatch tools, feed results back, respect
budgets and terminate? — with the model stubbed out.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_quality_agent.agent import AuditAgent
from data_quality_agent.llm import LLMResponse, ToolCall
from data_quality_agent.config import Settings
from data_quality_agent.memory import ELIDED, Transcript
from data_quality_agent.profiler import profile_dataframe
from data_quality_agent.tools import ToolBox


@pytest.fixture
def settings() -> Settings:
    return Settings(anthropic_api_key="sk-ant-test", max_steps=6, max_tool_calls=10)  # type: ignore[call-arg]


@pytest.fixture
def toolbox(sample_csv: Path, settings: Settings) -> ToolBox:
    profile = profile_dataframe(pd.read_csv(sample_csv), sample_csv.name)
    return ToolBox(sample_csv, profile, settings)


VALID_FINDING = {
    "title": "Age contains placeholder value 999",
    "category": "missing_data",
    "severity": "high",
    "columns": ["age"],
    "evidence": "300 of 5150 rows (5.8%) have age == 999, far outside the 18-84 range.",
    "why_it_matters": "999 will be treated as a real age and skew every model.",
    "recommendation": "Replace 999 with NaN, then impute or drop.",
}


# ---------- tool validation ----------

def test_valid_finding_is_recorded(toolbox: ToolBox):
    text, ok = toolbox.execute("record_finding", VALID_FINDING)
    assert ok
    assert len(toolbox.findings) == 1
    assert "Recorded finding #1" in text


def test_evidence_without_numbers_is_rejected(toolbox: ToolBox):
    bad = {**VALID_FINDING, "evidence": "there appear to be some odd values here"}
    text, ok = toolbox.execute("record_finding", bad)
    assert not ok
    assert toolbox.findings == []
    # The rejection must be actionable, so the model can retry successfully.
    assert "evidence" in text.lower()


def test_duplicate_findings_are_rejected(toolbox: ToolBox):
    toolbox.execute("record_finding", VALID_FINDING)
    text, ok = toolbox.execute("record_finding", {**VALID_FINDING, "title": "Different title"})
    assert not ok
    assert "Duplicate" in text
    assert len(toolbox.findings) == 1


def test_cannot_finish_with_no_findings(toolbox: ToolBox):
    text, ok = toolbox.execute(
        "finish_audit",
        {"overall_risk": "low", "summary": "Looks fine.", "ready_for_modeling": True},
    )
    assert not ok
    assert toolbox.finished is False


def test_unknown_tool_is_handled_gracefully(toolbox: ToolBox):
    text, ok = toolbox.execute("delete_everything", {})
    assert not ok
    assert "Unknown tool" in text


# ---------- memory ----------

def test_old_tool_results_are_elided_but_entries_survive():
    """Content is blanked; the result entries themselves must never be dropped."""
    t = Transcript(keep_full_results=2)
    for i in range(5):
        t.add_assistant("", [ToolCall(id=f"c{i}", name="run_pandas", arguments={})])
        t.add_tool_results(
            [{"id": f"c{i}", "name": "run_pandas", "content": f"BIG OUTPUT {i}", "is_error": False}]
        )

    msgs = t.for_api()

    call_ids = {c.id for m in msgs if m["role"] == "assistant" for c in m["tool_calls"]}
    result_ids = {r["id"] for m in msgs if m["role"] == "tool_results" for r in m["results"]}
    assert call_ids == result_ids  # every call still has an answer

    contents = [r["content"] for m in msgs if m["role"] == "tool_results" for r in m["results"]]
    assert contents[:3] == [ELIDED] * 3
    assert contents[3:] == ["BIG OUTPUT 3", "BIG OUTPUT 4"]


# ---------- the loop, with a fake model ----------

def _call(name, args):
    return ToolCall(id=f"c_{name}_{len(args)}", name=name, arguments=args)


class FakeProvider:
    """Replays a fixed script. Substitutes for a real model in tests."""

    name = "fake"
    model = "fake-1"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, system, messages, tools):
        self.calls += 1
        if self.script:
            text, calls = self.script.pop(0)
        else:
            text, calls = "done", []
        return LLMResponse(text=text, tool_calls=calls, input_tokens=100, output_tokens=50)

    def retryable(self):
        return ()


def test_full_loop_happy_path(sample_csv: Path, settings: Settings):
    fake = FakeProvider([
        ("", [_call("run_pandas", {"hypothesis": "age has 999s", "code": "result = (df['age'] == 999).sum()"})]),
        ("", [_call("record_finding", VALID_FINDING)]),
        ("", [_call("finish_audit", {
            "overall_risk": "high",
            "summary": "Placeholder values present.",
            "ready_for_modeling": False,
            "next_steps": ["Clean age column"],
        })]),
    ])
    agent = AuditAgent(settings, provider=fake)

    report = agent.audit(sample_csv, user_context="churn data, target=churned")

    assert report.status == "completed"
    assert len(report.findings) == 1
    assert report.summary is not None
    assert report.summary.ready_for_modeling is False
    assert report.steps_used == 3
    assert fake.calls == 3
    assert report.usage.input_tokens == 300
    assert len(report.trace) == 3
    assert "999" in report.to_markdown()


def test_loop_stops_at_step_budget(sample_csv: Path, settings: Settings):
    """A model that never calls finish_audit must not loop forever."""
    forever = [("", [_call("run_pandas", {"hypothesis": "h", "code": "result = 1"})]) for _ in range(50)]
    agent = AuditAgent(settings, provider=FakeProvider(forever))

    report = agent.audit(sample_csv)

    assert report.steps_used == settings.max_steps
    assert report.status == "completed"
    assert report.summary is None


def test_prose_only_response_is_nudged_not_fatal(sample_csv: Path, settings: Settings):
    agent = AuditAgent(settings, provider=FakeProvider([
        ("Sure, let me think about this dataset.", []),
        ("", [_call("record_finding", VALID_FINDING)]),
        ("", [_call("finish_audit", {
            "overall_risk": "medium", "summary": "ok", "ready_for_modeling": True})]),
    ]))

    report = agent.audit(sample_csv)
    assert report.status == "completed"
    assert len(report.findings) == 1


def test_unreadable_file_fails_cleanly(tmp_path: Path, settings: Settings):
    bad = tmp_path / "broken.csv"
    bad.write_bytes(b"\x00\x01\x02not a csv at all")
    report = AuditAgent(settings, provider=FakeProvider([])).audit(bad)
    assert report.status == "failed"
    assert report.error is not None
    assert report.findings == []


def test_sandbox_sees_exactly_the_rows_that_were_profiled(sample_csv: Path):
    """MAX_ROWS_SCANNED capped the profile but not the sandbox, so the agent was
    reasoning about a 100-row profile while measuring 5150 rows."""
    capped = Settings(  # type: ignore[call-arg]
        anthropic_api_key="sk-ant-test", max_steps=4, max_tool_calls=6, max_rows_scanned=100
    )
    fake = FakeProvider([
        ("", [_call("run_pandas", {"hypothesis": "how many rows", "code": "result = len(df)"})]),
        ("", [_call("record_finding", VALID_FINDING)]),
        ("", [_call("finish_audit", {
            "overall_risk": "low", "summary": "ok", "ready_for_modeling": True})]),
    ])

    report = AuditAgent(capped, provider=fake).audit(sample_csv)

    assert report.profile is not None and report.profile.n_rows == 100
    measured = next(t for t in report.trace if t.tool == "run_pandas")
    assert "100" in measured.output_preview
    assert "5150" not in measured.output_preview


def test_the_sandbox_copy_is_cleaned_up(sample_csv: Path, settings: Settings):
    """The copy holds user data; it must not outlive the run."""
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("dqa_*"))
    AuditAgent(settings, provider=FakeProvider([
        ("", [_call("finish_audit", {
            "overall_risk": "low", "summary": "ok", "ready_for_modeling": True})]),
    ])).audit(sample_csv)
    assert set(Path(tempfile.gettempdir()).glob("dqa_*")) == before


def test_sandbox_copy_falls_back_to_csv_when_parquet_refuses(
    sample_csv: Path, monkeypatch: pytest.MonkeyPatch
):
    """Messy frames are the ones parquet rejects, and messy frames are the job."""
    from data_quality_agent.profiler import cache_for_sandbox

    def boom(*_a, **_k):
        raise ValueError("cannot mix types in column 'x'")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    df = pd.read_csv(sample_csv, nrows=50)

    path = cache_for_sandbox(df, "run123")
    try:
        assert path is not None and path.suffix == ".csv"
        assert len(pd.read_csv(path)) == 50
    finally:
        if path:
            path.unlink(missing_ok=True)
