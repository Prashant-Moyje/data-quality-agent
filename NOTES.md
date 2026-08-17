# Private notes — not committed

This file is gitignored. It's prep material for talking about the project, not
documentation for anyone using it.

---

## Resume / portfolio description

> **Data Detective — Autonomous Data Quality Auditing Agent** · Python, Claude API, FastAPI, Streamlit, pandas, Pydantic
>
> Built a production-shaped LLM agent that autonomously audits tabular datasets: it profiles data deterministically, generates hypotheses about quality defects, writes and executes its own pandas code in a sandboxed subprocess, and produces a severity-ranked, evidence-backed report with a runnable cleaning script. Implemented the agent loop directly on the Anthropic tool-use API — no orchestration framework — with hard step/tool/timeout budgets and prompt caching for cost control. Hardened LLM-generated code execution behind a three-layer sandbox (AST allowlist, stripped-builtins subprocess, OS resource limits) and enforced output correctness with Pydantic validators that reject unsubstantiated findings and feed the error back to the model for self-correction. Validated against a synthetic dataset with nine planted defects; 40 tests covering sandbox escapes, budget enforcement and failure recovery run without network access.

**One-liner for a CV:** Autonomous agent that audits datasets by writing and sandbox-executing its own pandas code, returning evidence-backed, severity-ranked quality findings.

---

## Questions to be ready for

**"Why no LangChain?"** The loop is 60 lines. A framework would add a dependency tree and an abstraction to debug through, in exchange for hiding the part of the project that demonstrates I understand agents. I'd reach for LangGraph if I needed durable/resumable state or human-in-the-loop checkpoints.

**"How do you stop it hallucinating findings?"** Two mechanisms. The model never computes statistics — Python does, so numbers can't be invented. And a Pydantic validator rejects any finding whose evidence field contains no digits, with the rejection fed back so the model retries against real data.

**"What happens if the model writes malicious code?"** Three layers, and I assume each can fail — see the security section of the README. I'd add container isolation before putting this anywhere real.

**"How do you control cost?"** Hard step and tool-call caps; prompt caching on the system prompt; old tool outputs elided from the resent transcript; per-run token accounting exposed in the report.

**"How do you test something non-deterministic?"** Split the model out. The orchestration is tested against a fake client that replays scripted responses — that covers budgets, dispatch, error recovery and termination deterministically. Agent output quality is evaluated separately against the nine planted defects.

---

## Things worth mentioning unprompted

- The test suite caught a real bug in my own data generator: I was adding the
  unique `notes` column *after* concatenating duplicate rows, which silently
  destroyed all 150 planted duplicates. Good example of tests catching a
  fixture bug, not just a code bug.
- `memory.py` has a non-obvious constraint: you can't drop a `tool_result`
  block to save context, because the API requires every `tool_use` to have a
  matching result. So it elides the content and keeps the block.
- Findings live in `ToolBox`, not the transcript, specifically so context
  trimming can't destroy them. Context and memory are different things.
- Target leakage (`cancellation_reason` populated only when `churned == 1`) is
  the defect that separates a real audit from a null-count report.
