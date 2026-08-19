"""The agent loop.

WHY NO FRAMEWORK (LangChain / LangGraph / CrewAI)?
--------------------------------------------------
This agent is one while-loop with a tool dispatcher. A framework would add a
dependency tree, an abstraction to debug through, and version churn — in
exchange for hiding the ~60 lines that are the actual intellectual content of
the project. In an interview, "I wrote the loop" beats "I configured an
AgentExecutor" every time, because you can answer follow-ups about it.

Frameworks earn their keep at multi-agent orchestration, durable/resumable
state, and human-in-the-loop checkpoints. This project has none of those. If it
grew to need resumable runs, LangGraph would be the right migration.

The loop is also provider-agnostic: it talks to an LLMProvider (see llm.py) and
never learns whether the model is a local Ollama process or a hosted API.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import Settings, get_settings
from .llm import LLMError, LLMProvider, build_provider
from .logging_setup import get_logger
from .memory import Transcript
from .profiler import load_dataframe, profile_dataframe
from .schemas import AuditReport, DatasetProfile, ToolCallLog, Usage
from .tools import TOOL_SCHEMAS, ToolBox

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a meticulous senior data scientist performing a data \
quality audit. You have been handed an unfamiliar dataset and must determine \
whether it is fit for analysis or modeling.

METHOD
1. Read the profile you are given. It is computed deterministically and is
   trustworthy — never recompute what it already tells you.
2. From the profile, form specific, falsifiable hypotheses about what is wrong.
   Prioritise by damage: a leaking target column or a silently broken join
   matters more than a column with 2% nulls.
3. Test each hypothesis with run_pandas. Never assert anything you have not
   measured. If a query fails, read the error and rewrite it.
4. Record confirmed problems with record_finding, citing the numbers you saw.
5. Call finish_audit with your verdict.

WHAT A GOOD AUDITOR LOOKS FOR (beyond nulls and duplicates, which are obvious)
- Target leakage: a feature that could only be known after the label.
- Impossible values: negative ages/prices/durations, dates in the future,
  percentages above 100, end dates before start dates.
- Placeholder poison: 999, -1, 0, "N/A", "unknown", "" masquerading as real data.
- Type mismatch: numbers stored as strings, dates stored as text, booleans as
  "Yes"/"Y"/"true"/1 in the same column.
- Inconsistent categories: "NY" / "N.Y." / "new york" as separate levels.
- Cardinality traps: an ID column that will one-hot into 40,000 features; a
  "category" column with exactly one value (zero information).
- Class imbalance in a plausible target column.
- Duplicate entities that are not exact duplicate rows (same customer_id,
  different values).

RULES
- One hypothesis per run_pandas call. Aggregate your output; do not dump rows.
- Evidence must contain real numbers from a query you ran.
- Severity is about consequence, not about how interesting the bug is.
- Be terse. No preamble, no restating the schema back to the user.
- You have a limited step budget. Spend it on what would change a decision.
"""


class AuditAgent:
    """Runs one audit end to end."""

    def __init__(self, settings: Settings | None = None, provider: LLMProvider | None = None):
        self.settings = settings or get_settings()
        # Injectable so tests can pass a fake provider instead of a live model.
        self.provider = provider or build_provider(self.settings)

    # -- public API ------------------------------------------------------
    def audit(
        self,
        data_path: Path,
        user_context: str = "",
        on_progress: Callable[[str], None] | None = None,
    ) -> AuditReport:
        run_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        report = AuditReport(run_id=run_id, dataset_name=data_path.name)
        emit = on_progress or (lambda _msg: None)

        log.info("audit.start", run_id=run_id, dataset=data_path.name)

        # --- Phase 1: deterministic profiling (no LLM) ---
        try:
            emit("Profiling dataset...")
            df = load_dataframe(data_path, self.settings.max_rows_scanned)
            profile: DatasetProfile = profile_dataframe(df, data_path.name)
            report.profile = profile
            del df  # the agent reads data through the sandbox, not this process
        except Exception as e:
            log.exception("audit.profile_failed", run_id=run_id)
            report.status = "failed"
            report.error = f"Could not read dataset: {e}"
            return report

        toolbox = ToolBox(data_path, profile, self.settings)
        transcript = Transcript(keep_full_results=4)

        opening = profile.to_prompt_text()
        if user_context.strip():
            opening += f"\n\nCONTEXT FROM THE USER:\n{user_context.strip()}"
        opening += "\n\nBegin the audit."
        transcript.add_user(opening)

        # --- Phase 2: the agent loop ---
        try:
            self._loop(transcript, toolbox, report, emit)
        except LLMError as e:
            report.status = "failed"
            report.error = str(e)
        except Exception as e:
            log.exception("audit.crashed", run_id=run_id)
            report.status = "failed"
            report.error = f"Unexpected error: {type(e).__name__}: {e}"

        # --- Phase 3: assemble ---
        report.findings = toolbox.findings
        report.summary = toolbox.summary
        report.duration_seconds = round(time.perf_counter() - started, 2)
        if report.status != "failed":
            report.status = "completed"

        log.info(
            "audit.done",
            run_id=run_id,
            status=report.status,
            findings=len(report.findings),
            steps=report.steps_used,
        )
        return report

    # -- internals -------------------------------------------------------
    def _loop(
        self,
        transcript: Transcript,
        toolbox: ToolBox,
        report: AuditReport,
        emit: Callable[[str], None],
    ) -> None:
        s = self.settings

        for step in range(1, s.max_steps + 1):
            report.steps_used = step

            # Budget warning injected as a real message so the model can plan
            # its exit rather than being cut off mid-investigation.
            if step == s.max_steps - 1:
                transcript.add_user(
                    "BUDGET WARNING: this is your final investigation step. "
                    "Record any remaining findings and call finish_audit now."
                )

            emit(f"Step {step}/{s.max_steps}: thinking...")
            response = self._call_llm(transcript, report)
            transcript.add_assistant(response.text, response.tool_calls)

            tool_uses = response.tool_calls
            if not tool_uses:
                # Model replied with prose instead of a tool call. Nudge once.
                log.info("agent.no_tool_use", step=step, text=response.text[:200])
                transcript.add_user(
                    "You must act via tools. Call run_pandas to investigate, "
                    "record_finding to log a confirmed issue, or finish_audit to stop."
                )
                continue

            results = []
            for call in tool_uses:
                if toolbox.call_count >= s.max_tool_calls:
                    results.append(
                        self._tool_result(
                            call, "Tool budget exhausted. Call finish_audit immediately.", True
                        )
                    )
                    continue

                emit(f"Step {step}: {call.name}")
                text, ok = toolbox.execute(call.name, call.arguments)
                report.trace.append(
                    ToolCallLog(
                        step=step,
                        tool=call.name,
                        input=call.arguments,
                        ok=ok,
                        output_preview=text[:300],
                    )
                )
                results.append(self._tool_result(call, text, is_error=not ok))

            transcript.add_tool_results(results)

            if toolbox.finished:
                emit("Audit complete.")
                return

        log.warning("agent.step_limit", steps=s.max_steps)

    def _call_llm(self, transcript: Transcript, report: AuditReport):
        """One provider call, with bounded retry on transient failures."""
        retryable = getattr(self.provider, "retryable", lambda: ())()
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                resp = self.provider.complete(
                    system=SYSTEM_PROMPT,
                    messages=transcript.for_api(),
                    tools=TOOL_SCHEMAS,
                )
                self._track_usage(resp, report)
                return resp
            except retryable as e:  # type: ignore[misc]
                last_exc = e
                wait = 2 ** attempt
                log.warning("llm.retry", attempt=attempt + 1, wait=wait, error=str(e)[:200])
                time.sleep(wait)

        raise LLMError(f"Provider failed after 3 attempts: {last_exc}")

    def _track_usage(self, resp: Any, report: AuditReport) -> None:
        u: Usage = report.usage
        u.input_tokens += resp.input_tokens
        u.output_tokens += resp.output_tokens

        if self.settings.provider == "ollama":
            u.estimated_cost_usd = 0.0  # local inference is free
            return

        pi, po = self.settings.price_per_mtok_input, self.settings.price_per_mtok_output
        if pi or po:
            u.estimated_cost_usd = round(
                (u.input_tokens / 1e6) * pi + (u.output_tokens / 1e6) * po, 4
            )

    @staticmethod
    def _tool_result(call, text: str, is_error: bool = False) -> dict[str, Any]:
        return {"id": call.id, "name": call.name, "content": text, "is_error": is_error}
