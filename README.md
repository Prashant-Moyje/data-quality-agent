#  Ground Truth

**An autonomous agent that audits a dataset the way a senior data scientist would: it forms hypotheses about what's broken, writes its own pandas code to test them, runs that code in a sandbox, and reports only what it can prove with numbers.**

You give it a CSV. It gives you a severity-ranked list of data quality problems, each backed by evidence it actually measured  plus a runnable cleaning script.

```bash
ground-truth data/messy_customers.csv --context "Customer churn export. Target column is churned."
```

---

## Engineering decisions

| Problem | Solution in this repo |
|---|---|
| LLMs are bad at arithmetic | `profiler.py` computes every statistic in Python. The model never does math, only interpretation. |
| LLM-generated code is remote code execution | `sandbox.py`  AST allowlist + isolated subprocess + OS resource limits. |
| Models assert things they didn't verify | Pydantic validator rejects any finding whose evidence contains no numbers. |
| Agents loop forever / burn money | Hard step, tool-call, timeout and memory budgets, with a "final step" warning injected so the model exits gracefully. |
| Stateless API = quadratic token cost | `memory.py` elides old tool output while preserving `tool_use`/`tool_result` pairing. |
| Findings must survive context trimming | Findings live in `ToolBox`, not the transcript. Context  memory. |

---

## Architecture

```
                    +--------------+
   CSV upload  ---> |  profiler.py |  deterministic facts (no LLM)
                    +------+-------+
                           |
                           v
        +--------------------------------------+
        |            agent.py loop             |
        |                                      |
        |    LLM  ---> picks a tool ---+       |
        |     ^                        |       |
        |     |                        v       |
        |     |              +----------------+|
        |     |              |  run_pandas    |+---> sandbox.py ---> subprocess
        |     |              | record_finding |+---> Pydantic validation
        |     |              |  finish_audit  ||
        |     |              +--------+-------+|
        |     +--- tool result
```

**Flow:** profile deterministically  agent hypothesises  sandboxed execution  validated findings  structured report.

---

## Tech choices, and why

| Choice | Why | What I rejected |
|---|---|---|
| **Claude Sonnet 5** via the official `anthropic` SDK | Strong at tool use and code generation, which is the entire job here. Native prompt caching cuts cost on the resent system prompt. |  |
| **No agent framework** | The loop is ~60 lines and is the intellectual content of the project. A framework would hide it behind an abstraction I'd then have to debug. | LangChain, LangGraph, CrewAI. LangGraph would be the right call *if* this needed resumable/durable runs or human approval gates. It doesn't. |
| **Pydantic v2** | Turns "the model returned something weird" from a silent corruption into a typed error I can feed back to the model so it self-corrects. | Parsing JSON by hand |
| **Subprocess + AST allowlist** | Defence in depth against prompt injection hidden inside the data itself. | `exec()` in-process (RCE), full Docker-per-call (too slow for a demo) |
| **FastAPI + background tasks** | Audits take 3090s. Async job + polling is the standard shape for agent-backed APIs; a sync endpoint would hit proxy timeouts. | Sync endpoint, Celery (infrastructure with nothing to prove) |
| **Streamlit as a thin API client** | Holds zero agent logic, which proves the backend is genuinely reusable. | Streamlit calling the agent directly |
| **structlog** | Agent runs are non-deterministic and multi-step. You need to grep by `run_id` and step. | `print()`, stdlib f-string logging |
| **In-memory dict + JSON files** | Single-node demo. Swapping in Postgres is a repo change, not a redesign. | Postgres/Redis for a portfolio project |

---

## Setup

**Requires Python 3.10+ and [Ollama](https://ollama.com/download). No API key, no account, no cost - the model runs on your machine.**

```bash
git clone https://github.com/Prashant-Moyje/ground-truth.git
cd ground-truth

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

ollama pull qwen3:8b                  # ~5 GB, one time

cp .env.example .env                  # Windows cmd: copy .env.example .env
# defaults are already set for local Ollama - nothing to edit

python scripts/make_sample_data.py    # generates data/messy_customers.csv
```
## Run with Docker

The whole stack — Ollama, API, and UI — comes up with one command:

```bash
docker compose up --build
docker compose exec ollama ollama pull qwen3:8b   # first run only, ~5 GB
```

UI at `http://localhost:8501`, API docs at `http://localhost:8000/docs`.

Requires ~8 GB allocated to Docker (Settings → Resources → Memory).

### Why the container matters for security

The `api` service runs with `read_only: true`, `cap_drop: ALL`, `no-new-privileges`, a 2 GB memory cap, and `pids_limit: 256`.

This closes a real gap. The OS-level limits in `_runner.py` use `RLIMIT_AS` and `RLIMIT_CPU`, which **do not exist on Windows** — that layer is inert when the app runs natively there. Inside the container those limits are enforced by the kernel regardless of host OS. Docker isn't packaging convenience here; it makes a security layer real that was otherwise decorative.

### No hosted demo

Running the agent needs a local LLM with ~8 GB RAM, and an audit takes tens of minutes on CPU. Free hosting tiers can't supply either, so a public URL would show a sleeping or timing-out app rather than a working one. The CLI output and evaluation scorecard above are the honest demo.
### Run it  three ways

**1. CLI (fastest to demo)**
```bash
ground-truth data/messy_customers.csv \
  --context "Customer churn export from our CRM. Target is churned. One row per customer." \
  --out report.md --fix-script cleanup.py
```

**2. API**
```bash
uvicorn ground_truth.api:app --reload
# docs at http://127.0.0.1:8000/docs

curl -X POST http://127.0.0.1:8000/audits \
  -F "file=@data/messy_customers.csv" \
  -F "context=Churn data, target=churned"
# -> {"run_id":"a1b2c3d4e5f6","status":"running"}

curl http://127.0.0.1:8000/audits/a1b2c3d4e5f6
```

**3. Full UI** (two terminals)
```bash
uvicorn ground_truth.api:app          # terminal 1
streamlit run src/ground_truth/ui.py  # terminal 2 -> localhost:8501
```

### Tests
```bash
pytest              # 40 tests, no API key or network required
pytest --cov=src
```
The agent tests use a fake Anthropic client. You cannot write reliable tests against a non-deterministic model, so the orchestration is tested with the model stubbed out  budgets, tool dispatch, error recovery, termination.

---

## The evaluation dataset

`scripts/make_sample_data.py` plants **nine known defects**, so you can measure whether the agent actually found things rather than generated plausible prose:

1. Placeholder poison  `age == 999` (300 rows)
2. Impossible values  `tenure_months == -1` (120 rows)
3. Real missingness  `monthly_charge` null (450 rows)
4. Inconsistent categories  `ny` / `N.Y.` / `New York` / `calif.`
5. Type mismatch  `last_payment` stored as `"$70.12"` strings
6. Zero variance  `data_source` has one value
7. **Target leakage**  `cancellation_reason` populated only when `churned == 1`
8. Duplicates  150 exact duplicate rows
9. High cardinality  `notes` unique per row

Defect 7 is the interesting one. Nulls and duplicates are table stakes; catching that a feature is only ever populated for the positive class is the thing a junior analyst misses and a leaked model dies on.

---

## Security posture (read this before deploying)

The agent executes code it wrote itself. If a CSV contains a column named `__import__('os').system(...)`, naive `exec()` is remote code execution. Three independent layers:

1. **Static**  AST allowlist rejects imports, dunder access, `eval`/`exec`/`open`/`getattr`, and pandas' disk/network methods.
2. **Runtime**  separate `python -I` subprocess with stripped `__builtins__`, so a bypass lands in a crippled interpreter, not the app's.
3. **Resource**  `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE=0` (cannot write files at all), plus a wall-clock timeout.

Also: uploads are size-capped and streamed to disk, the user's filename is never used as a path, and the temp file is deleted after the run.

**This is not a true sandbox.** Escaping CPython restricted execution is a known research sport. For production, run the subprocess in a container with no network namespace and a read-only filesystem  the code is already isolated in `_runner.py` specifically so that swap is a deployment change, not a rewrite.

---

## Project structure

```
ground-truth/
├── src/ground_truth/
│   ├── __init__.py
│   ├── config.py         # fail-fast settings from .env
│   ├── logging_setup.py  # structlog
│   ├── schemas.py        # Pydantic contracts + report rendering
│   ├── profiler.py       # deterministic stats — the agent's ground truth
│   ├── sandbox.py        # AST validation + isolated execution   ← security core
│   ├── _runner.py        # the sandbox child process (never imported by app code)
│   ├── llm.py            # provider abstraction: Ollama / Anthropic
│   ├── tools.py          # tool schemas + dispatch + findings ledger
│   ├── memory.py         # provider-neutral transcript + trimming
│   ├── agent.py          # the loop                              ← logic core
│   ├── api.py            # FastAPI backend
│   ├── ui.py             # Streamlit front end
│   └── cli.py            # command-line entry point
├── scripts/
│   ├── make_sample_data.py   # generates the 9-defect evaluation dataset
│   └──

```
---

## Known limitations

- Single-node storage; restarting the API loses in-flight runs.
- Loads the dataset into memory  capped at 500k rows by default. Larger data would need sampling or a DuckDB backend.
- Findings are LLM judgements. The evidence is real (it was measured), but severity is an opinion.
- No cross-run memory. Auditing the same table twice repeats the work.

---

## License

MIT
