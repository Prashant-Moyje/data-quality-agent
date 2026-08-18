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
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

import anthropic

from .config import Settings, get_settings
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

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

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
        except anthropic.AuthenticationError:
            report.status = "failed"
            report.error = "Anthropic API key is invalid or missing."
        except anthropic.APIError as e:
            log.exception("audit.api_error", run_id=run_id)
            report.status = "failed"
            report.error = f"LLM API error: {e}"
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

            transcript.add_assistant(
                [b.model_dump(exclude_none=True) for b in response.content]
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # Model replied with prose instead of a tool call. Nudge once.
                text = " ".join(b.text for b in response.content if b.type == "text")
                log.info("agent.no_tool_use", step=step, text=text[:200])
                transcript.add_user(
                    "You must act via tools. Call run_pandas to investigate, "
                    "record_finding to log a confirmed issue, or finish_audit to stop."
                )
                continue

            results = []
            for block in tool_uses:
                if toolbox.call_count >= s.max_tool_calls:
                    results.append(
                        self._tool_result(
                            block.id,
                            "Tool budget exhausted. Call finish_audit immediately.",
                            is_error=True,
                        )
                    )
                    continue

                emit(f"Step {step}: {block.name}")
                text, ok = toolbox.execute(block.name, dict(block.input))
                report.trace.append(
                    ToolCallLog(
                        step=step,
                        tool=block.name,
                        input=dict(block.input),
                        ok=ok,
                        output_preview=text[:300],
                    )
                )
                results.append(self._tool_result(block.id, text, is_error=not ok))

            transcript.add_tool_results(results)

            if toolbox.finished:
                emit("Audit complete.")
                return

        log.warning("agent.step_limit", steps=s.max_steps)

    def _call_llm(self, transcript: Transcript, report: AuditReport):
        """One API call, with bounded retry on transient failures."""
        s = self.settings
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=s.model,
                    max_tokens=s.max_tokens_per_call,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            # Prompt caching: the system prompt is resent on every
                            # one of the ~14 turns. Caching it cuts input cost
                            # substantially for a one-line change.
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=TOOL_SCHEMAS,
                    messages=transcript.for_api(),
                )
                self._track_usage(resp, report)
                return resp
            except (anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.InternalServerError) as e:
                last_exc = e
                wait = 2 ** attempt
                log.warning("llm.retry", attempt=attempt + 1, wait=wait, error=str(e)[:200])
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    def _track_usage(self, resp: Any, report: AuditReport) -> None:
        u: Usage = report.usage
        u.input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
        u.output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
        u.cache_read_tokens += getattr(resp.usage, "cache_read_input_tokens", 0) or 0

        pi, po = self.settings.price_per_mtok_input, self.settings.price_per_mtok_output
        if pi or po:
            u.estimated_cost_usd = round(
                (u.input_tokens / 1e6) * pi + (u.output_tokens / 1e6) * po, 4
            )

    @staticmethod
    def _tool_result(tool_use_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": text,
        }
        if is_error:
            block["is_error"] = True
        return block
