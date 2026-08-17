"""Tool, memory and agent-loop tests.

The agent test uses a FAKE Anthropic client. This is the important pattern: you
cannot write reliable tests against a non-deterministic model, so you test the
*orchestration* — does the loop dispatch tools, feed results back, respect
budgets and terminate? — with the model stubbed out.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from data_detective.agent import AuditAgent
from data_detective.config import Settings
from data_detective.memory import ELIDED, Transcript
from data_detective.profiler import profile_dataframe
from data_detective.tools import ToolBox


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

def test_old_tool_results_are_elided_but_blocks_survive():
    """Deleting a tool_result block would be an API 400. We elide content only."""
    t = Transcript(keep_full_results=2)
    for i in range(5):
        t.add_assistant([{"type": "tool_use", "id": f"tu_{i}", "name": "run_pandas", "input": {}}])
        t.add_tool_results([{"type": "tool_result", "tool_use_id": f"tu_{i}", "content": f"BIG OUTPUT {i}"}])

    api_msgs = t.for_api()

    tool_use_ids = {
        b["id"] for m in api_msgs for b in m["content"] if b.get("type") == "tool_use"
    }
    tool_result_ids = {
        b["tool_use_id"] for m in api_msgs for b in m["content"] if b.get("type") == "tool_result"
    }
    assert tool_use_ids == tool_result_ids  # every call still has an answer

    contents = [
        b["content"] for m in api_msgs for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert contents[:3] == [ELIDED] * 3     # old ones elided
    assert contents[3:] == ["BIG OUTPUT 3", "BIG OUTPUT 4"]  # recent ones intact


# ---------- the loop, with a fake model ----------

def _blocks(*specs):
    out = []
    for kind, payload in specs:
        if kind == "text":
            out.append(SimpleNamespace(type="text", text=payload,
                                       model_dump=lambda exclude_none=True, p=payload: {"type": "text", "text": p}))
        else:
            name, tool_input = payload
            tid = f"tu_{id(payload)}"
            out.append(SimpleNamespace(
                type="tool_use", id=tid, name=name, input=tool_input,
                model_dump=lambda exclude_none=True, t=tid, n=name, i=tool_input: {
                    "type": "tool_use", "id": t, "name": n, "input": i},
            ))
    return out


class FakeMessages:
    """Replays a fixed script of model responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        content = self.script.pop(0) if self.script else _blocks(("text", "done"))
        return SimpleNamespace(
            content=content,
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0),
        )


def test_full_loop_happy_path(sample_csv: Path, settings: Settings, monkeypatch):
    script = [
        _blocks(("tool", ("run_pandas", {"hypothesis": "age has 999s", "code": "result = (df['age'] == 999).sum()"}))),
        _blocks(("tool", ("record_finding", VALID_FINDING))),
        _blocks(("tool", ("finish_audit", {
            "overall_risk": "high",
            "summary": "Placeholder values present.",
            "ready_for_modeling": False,
            "next_steps": ["Clean age column"],
        }))),
    ]
    agent = AuditAgent(settings)
    fake = FakeMessages(script)
    monkeypatch.setattr(agent.client, "messages", fake)

    report = agent.audit(sample_csv, user_context="churn data, target=churned")

    assert report.status == "completed"
    assert len(report.findings) == 1
    assert report.summary is not None
    assert report.summary.ready_for_modeling is False
    assert report.steps_used == 3          # stopped as soon as finish_audit was called
    assert fake.calls == 3                 # no wasted API calls
    assert report.usage.input_tokens == 300
    assert len(report.trace) == 3          # full audit trail
    assert "999" in report.to_markdown()


def test_loop_stops_at_step_budget(sample_csv: Path, settings: Settings, monkeypatch):
    """A model that never calls finish_audit must not loop forever."""
    forever = [
        _blocks(("tool", ("run_pandas", {"hypothesis": "h", "code": "result = 1"})))
        for _ in range(50)
    ]
    agent = AuditAgent(settings)
    monkeypatch.setattr(agent.client, "messages", FakeMessages(forever))

    report = agent.audit(sample_csv)

    assert report.steps_used == settings.max_steps
    assert report.status == "completed"  # degraded, not crashed
    assert report.summary is None


def test_prose_only_response_is_nudged_not_fatal(sample_csv: Path, settings: Settings, monkeypatch):
    script = [
        _blocks(("text", "Sure, let me think about this dataset.")),
        _blocks(("tool", ("record_finding", VALID_FINDING))),
        _blocks(("tool", ("finish_audit", {
            "overall_risk": "medium", "summary": "ok", "ready_for_modeling": True}))),
    ]
    agent = AuditAgent(settings)
    monkeypatch.setattr(agent.client, "messages", FakeMessages(script))

    report = agent.audit(sample_csv)
    assert report.status == "completed"
    assert len(report.findings) == 1


def test_unreadable_file_fails_cleanly(tmp_path: Path, settings: Settings):
    bad = tmp_path / "broken.csv"
    bad.write_bytes(b"\x00\x01\x02not a csv at all")
    report = AuditAgent(settings).audit(bad)
    assert report.status == "failed"
    assert report.error is not None
    assert report.findings == []  # no hallucinated output on failure
