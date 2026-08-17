"""The agent's action space.

A tool schema is a prompt. Vague descriptions produce vague tool calls, so each
one below states *when* to use it and what good input looks like. Four tools is
deliberate — every extra tool measurably degrades selection accuracy, and this
job genuinely only needs: look, test, write down, stop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import Settings
from .logging_setup import get_logger
from .sandbox import run_snippet
from .schemas import AuditSummary, DatasetProfile, Finding

log = get_logger(__name__)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "run_pandas",
        "description": (
            "Execute a read-only pandas snippet against the dataset to test a "
            "hypothesis. The DataFrame is already loaded as `df`; `pd` and `np` "
            "are in scope. Do NOT import or read files. Assign what you want to "
            "see to a variable named `result`, or use print(). Output is "
            "truncated to ~4000 characters, so aggregate rather than dumping raw "
            "rows. This is your only way to see actual data — use it before every "
            "finding you record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": "What you are testing, in one sentence. e.g. "
                                   "'signup_date may contain dates after churn_date'.",
                },
                "code": {
                    "type": "string",
                    "description": "Python. e.g. "
                                   "result = (df['age'] < 0).sum()",
                },
            },
            "required": ["hypothesis", "code"],
        },
    },
    {
        "name": "record_finding",
        "description": (
            "Record a confirmed data-quality problem. Only call this AFTER "
            "run_pandas has produced numeric evidence for it — `evidence` must "
            "cite real observed numbers and will be rejected if it does not. "
            "One finding per distinct problem; do not record the same issue twice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "missing_data", "duplicates", "outliers",
                        "inconsistent_format", "type_mismatch", "target_leakage",
                        "class_imbalance", "cardinality", "temporal",
                        "logical_inconsistency",
                    ],
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                    "description": "critical = will break a model or mislead a "
                                   "business decision. low = cosmetic.",
                },
                "columns": {"type": "array", "items": {"type": "string"}},
                "evidence": {
                    "type": "string",
                    "description": "Observed numbers, e.g. '318/5000 rows (6.4%) "
                                   "have age < 0, min = -12'.",
                },
                "why_it_matters": {"type": "string"},
                "recommendation": {"type": "string"},
                "fix_code": {
                    "type": "string",
                    "description": "Optional pandas that fixes it, operating on `df`.",
                },
            },
            "required": [
                "title", "category", "severity", "evidence",
                "why_it_matters", "recommendation",
            ],
        },
    },
    {
        "name": "finish_audit",
        "description": (
            "End the audit and deliver the verdict. Call this when you have "
            "investigated the material risks, or when further investigation would "
            "not change the conclusion. Do not pad the audit with trivial checks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "overall_risk": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                },
                "summary": {
                    "type": "string",
                    "description": "2-4 sentences a data lead can act on.",
                },
                "ready_for_modeling": {"type": "boolean"},
                "next_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["overall_risk", "summary", "ready_for_modeling"],
        },
    },
]


class ToolBox:
    """Executes tool calls and owns the findings ledger.

    Keeping findings HERE rather than in the chat transcript matters: the
    transcript gets trimmed to control cost, but findings must survive. This is
    the difference between "context window" and "memory".
    """

    def __init__(self, data_path: Path, profile: DatasetProfile, settings: Settings):
        self.data_path = data_path
        self.profile = profile
        self.settings = settings
        self.findings: list[Finding] = []
        self.summary: AuditSummary | None = None
        self.finished = False
        self.call_count = 0

    # --- dispatch -------------------------------------------------------
    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Return (text_for_model, ok). Never raises: the model must see errors."""
        self.call_count += 1
        handler = {
            "run_pandas": self._run_pandas,
            "record_finding": self._record_finding,
            "finish_audit": self._finish_audit,
        }.get(name)

        if handler is None:
            return f"Unknown tool {name!r}. Available: run_pandas, record_finding, finish_audit.", False

        try:
            return handler(tool_input)
        except Exception as e:  # last-resort guard
            log.exception("tool.crashed", tool=name)
            return f"Tool crashed: {type(e).__name__}: {e}", False

    # --- handlers -------------------------------------------------------
    def _run_pandas(self, ti: dict[str, Any]) -> tuple[str, bool]:
        code = ti.get("code", "")
        if not code.strip():
            return "No code provided.", False

        log.info("tool.run_pandas", hypothesis=ti.get("hypothesis", "")[:120])
        res = run_snippet(
            code,
            self.data_path,
            timeout_s=self.settings.sandbox_timeout_s,
            memory_mb=self.settings.sandbox_memory_mb,
        )
        return res.to_tool_text(), res.ok

    def _record_finding(self, ti: dict[str, Any]) -> tuple[str, bool]:
        try:
            finding = Finding.model_validate(ti)
        except ValidationError as e:
            # Feeding the validation error back is what makes the agent
            # self-correct instead of failing the run.
            return f"Finding rejected by validation:\n{e}\nFix the fields and retry.", False

        # Cheap dedupe: same category + same column set.
        key = (finding.category, tuple(sorted(finding.columns)))
        for existing in self.findings:
            if (existing.category, tuple(sorted(existing.columns))) == key:
                return (
                    f"Duplicate: already recorded {existing.title!r} for these "
                    f"columns. Investigate something else."
                ), False

        self.findings.append(finding)
        log.info("tool.finding", title=finding.title, severity=finding.severity.value)
        return (
            f"Recorded finding #{len(self.findings)}: {finding.title} "
            f"[{finding.severity.value}]"
        ), True

    def _finish_audit(self, ti: dict[str, Any]) -> tuple[str, bool]:
        if not self.findings:
            # Guard against a lazy agent stopping before doing any work.
            return (
                "Cannot finish with zero findings recorded. Either investigate "
                "further, or record at least one finding describing the dataset's "
                "clean state before finishing."
            ), False
        try:
            self.summary = AuditSummary.model_validate(ti)
        except ValidationError as e:
            return f"Summary rejected:\n{e}\nRetry with corrected fields.", False

        self.finished = True
        log.info("tool.finish", risk=self.summary.overall_risk.value)
        return "Audit closed.", True
