"""Typed contracts for everything the agent produces.

WHY THIS FILE EXISTS
--------------------
An LLM will happily return prose. Prose is unusable downstream. Every tool the
agent can call has a Pydantic model here, so a malformed tool call is rejected
with a clear error *that we feed back to the model* instead of crashing the run
or silently producing garbage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    CRITICAL = "critical"  # will break a model or mislead a decision
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"  # cosmetic / worth knowing


class Category(str, Enum):
    MISSING_DATA = "missing_data"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMAT = "inconsistent_format"
    TYPE_MISMATCH = "type_mismatch"
    TARGET_LEAKAGE = "target_leakage"
    CLASS_IMBALANCE = "class_imbalance"
    CARDINALITY = "cardinality"
    TEMPORAL = "temporal"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"


class Finding(BaseModel):
    """One concrete, evidence-backed problem in the dataset."""

    title: str = Field(..., max_length=120, description="Short headline.")
    category: Category
    severity: Severity
    columns: list[str] = Field(default_factory=list)
    evidence: str = Field(
        ...,
        max_length=2000,
        description="Concrete numbers observed, e.g. '318/5000 rows (6.4%) have age < 0'.",
    )
    why_it_matters: str = Field(..., max_length=1000)
    recommendation: str = Field(..., max_length=1000)
    fix_code: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional pandas snippet that fixes it, operating on `df`.",
    )

    @field_validator("evidence")
    @classmethod
    def evidence_must_contain_a_number(cls, v: str) -> str:
        # Cheap but effective guard against vague, hallucinated findings like
        # "there appear to be some missing values". We force the model to cite data.
        if not any(ch.isdigit() for ch in v):
            raise ValueError(
                "evidence must cite concrete numbers observed via run_pandas "
                "(counts, percentages, or ranges)"
            )
        return v


class AuditSummary(BaseModel):
    """The agent's closing statement."""

    overall_risk: Severity
    summary: str = Field(..., max_length=2000)
    ready_for_modeling: bool
    next_steps: list[str] = Field(default_factory=list, max_length=10)


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[str]
    numeric_stats: dict[str, float] | None = None


class DatasetProfile(BaseModel):
    """Deterministic facts. Computed in Python, never by the LLM."""

    name: str
    n_rows: int
    n_cols: int
    memory_mb: float
    exact_duplicate_rows: int
    columns: list[ColumnProfile]

    def to_prompt_text(self) -> str:
        lines = [
            f"DATASET: {self.name}",
            f"shape: {self.n_rows} rows x {self.n_cols} columns | "
            f"{self.memory_mb:.1f} MB | exact duplicate rows: {self.exact_duplicate_rows}",
            "",
            "COLUMNS:",
        ]
        for c in self.columns:
            bits = [
                f"- {c.name} ({c.dtype})",
                f"nulls={c.null_count} ({c.null_pct:.1f}%)",
                f"unique={c.unique_count}",
            ]
            if c.numeric_stats:
                s = c.numeric_stats
                bits.append(
                    f"min={s['min']:.4g} p50={s['p50']:.4g} max={s['max']:.4g} mean={s['mean']:.4g}"
                )
            bits.append(f"e.g. {', '.join(c.sample_values[:4])}")
            lines.append(" | ".join(bits))
        return "\n".join(lines)


class ToolCallLog(BaseModel):
    step: int
    tool: str
    input: dict[str, Any]
    ok: bool
    output_preview: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_cost_usd: float | None = None


class AuditReport(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"] = "running"
    dataset_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    profile: DatasetProfile | None = None
    findings: list[Finding] = Field(default_factory=list)
    summary: AuditSummary | None = None
    trace: list[ToolCallLog] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    steps_used: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    def to_markdown(self) -> str:
        """Human-readable report for the UI / a GitHub screenshot."""
        out = [f"# Data Audit — {self.dataset_name}", ""]
        if self.profile:
            out += [
                f"**{self.profile.n_rows:,} rows x {self.profile.n_cols} columns**  ",
                f"Run `{self.run_id}` · {self.steps_used} agent steps · "
                f"{self.duration_seconds:.1f}s · "
                f"{self.usage.input_tokens + self.usage.output_tokens:,} tokens",
                "",
            ]
        if self.summary:
            out += [
                f"## Verdict: {self.summary.overall_risk.value.upper()} risk",
                "",
                f"Ready for modeling: **{'yes' if self.summary.ready_for_modeling else 'no'}**",
                "",
                self.summary.summary,
                "",
            ]
        out += [f"## Findings ({len(self.findings)})", ""]
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        for f in sorted(self.findings, key=lambda x: order[x.severity]):
            cols = f", `{'`, `'.join(f.columns)}`" if f.columns else ""
            out += [
                f"### [{f.severity.value.upper()}] {f.title}",
                f"*{f.category.value}{cols}*",
                "",
                f"**Evidence.** {f.evidence}",
                "",
                f"**Why it matters.** {f.why_it_matters}",
                "",
                f"**Fix.** {f.recommendation}",
                "",
            ]
            if f.fix_code:
                out += ["```python", f.fix_code, "```", ""]
        if self.summary and self.summary.next_steps:
            out += ["## Next steps", ""]
            out += [f"{i}. {s}" for i, s in enumerate(self.summary.next_steps, 1)]
        return "\n".join(out)

    def to_fix_script(self) -> str:
        """Concatenate every fix_code into one runnable cleaning script."""
        parts = [
            '"""Auto-generated cleaning script from Data Detective.',
            "",
            "REVIEW BEFORE RUNNING. These are suggestions, not decisions.",
            '"""',
            "import pandas as pd",
            "",
            f"df = pd.read_csv({self.dataset_name!r})",
            "",
        ]
        for f in self.findings:
            if f.fix_code:
                parts += [f"# [{f.severity.value}] {f.title}", f.fix_code, ""]
        parts += ["df.to_csv('cleaned.csv', index=False)"]
        return "\n".join(parts)
