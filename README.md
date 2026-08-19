# 🔎 Data Detective

**An autonomous agent that audits a dataset the way a senior data scientist would: it forms hypotheses about what's broken, writes its own pandas code to test them, runs that code in a sandbox, and reports only what it can prove with numbers.**

You give it a CSV. It gives you a severity-ranked list of data quality problems, each backed by evidence it actually measured — plus a runnable cleaning script.

```bash
data-detective data/messy_customers.csv --context "Customer churn export. Target column is churned."
```

---

## Engineering decisions

| Problem | Solution in this repo |
|---|---|
| LLMs are bad at arithmetic | `profiler.py` computes every statistic in Python. The model never does math, only interpretation. |
| LLM-generated code is remote code execution | `sandbox.py` — AST allowlist + isolated subprocess + OS resource limits. |
| Models assert things they didn't verify | Pydantic validator rejects any finding whose evidence contains no numbers. |
| Agents loop forever / burn money | Hard step, tool-call, timeout and memory budgets, with a "final step" warning injected so the model exits gracefully. |
| Stateless API = quadratic token cost | `memory.py` elides old tool output while preserving `tool_use`/`tool_result` pairing. |
| Findings must survive context trimming | Findings live in `ToolBox`, not the transcript. Context ≠ memory. |

---

## Architecture

```
                    ┌──────────────┐
   CSV upload  ───▶ │  profiler.py │  deterministic facts (no LLM)
                    └──────┬───────┘
                           ▼
        ┌──────────────────────────────────────┐
        │            agent.py loop             │
        │                                      │
        │   Claude ──▶ picks a tool ──┐        │
        │      ▲                      ▼        │
        │      │              ┌──────────────┐ │
        │      │              │  run_pandas  │─┼─▶ sandbox.py ─▶ subprocess
        │      │              │record_finding│─┼─▶ Pydantic validation
        │      │              │ finish_audit │ │
        │      │              └──────┬───────┘ │
        │      └── tool result ◀─────┘         │
        │           (budget-capped)            │
        └──────────────────┬───────────────────┘
                           ▼
              AuditReport  →  markdown report + fix_script.py
```

**Flow:** profile deterministically → agent hypothesises → sandboxed execution → validated findings → structured report.

---

## Tech choices, and why

| Choice | Why | What I rejected |
|---|---|---|
| **Pluggable LLM provider** (`llm.py`) — local Ollama by default, Claude optional | Anyone can clone and run this without an account or a bill, which matters more for a public repo than raw model quality. The abstraction also forced the loop to stay provider-agnostic. | Hardcoding one vendor. The two APIs disagree on tool schemas, tool-call IDs, tool-result message shape and token fields — normalising that is the interesting part. |
| **`qwen3:8b` as the local default** | Best tool-calling reliability per GB at the 8B size. Tool calling is the entire job here, and most small models are poor at it. | Larger local models (RAM-prohibitive), 3B models (drop tool calls under a long transcript) |
| **No agent framework** | The loop is ~60 lines and is the intellectual content of the project. A framework would hide it behind an abstraction I'd then have to debug. | LangChain, LangGraph, CrewAI. LangGraph would be the right call *if* this needed resumable/durable runs or human approval gates. It doesn't. |
| **Pydantic v2** | Turns "the model returned something weird" from a silent corruption into a typed error I can feed back to the model so it self-corrects. | Parsing JSON by hand |
| **Subprocess + AST allowlist** | Defence in depth against prompt injection hidden inside the data itself. | `exec()` in-process (RCE), full Docker-per-call (too slow for a demo) |
| **FastAPI + background tasks** | Audits take 30–90s. Async job + polling is the standard shape for agent-backed APIs; a sync endpoint would hit proxy timeouts. | Sync endpoint, Celery (infrastructure with nothing to prove) |
| **Streamlit as a thin API client** | Holds zero agent logic, which proves the backend is genuinely reusable. | Streamlit calling the agent directly |
| **structlog** | Agent runs are non-deterministic and multi-step. You need to grep by `run_id` and step. | `print()`, stdlib f-string logging |
| **In-memory dict + JSON files** | Single-node demo. Swapping in Postgres is a repo change, not a redesign. | Postgres/Redis for a portfolio project |

---

## Setup

**Requires Python 3.10+ and [Ollama](https://ollama.com/download). No API key, no account, no cost — the model runs on your machine.**

```bash
git clone https://github.com/Prashant-Moyje/data-detective.git
cd data-detective

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

ollama pull qwen3:8b                  # ~5 GB, one time

cp .env.example .env                  # Windows: copy .env.example .env
# defaults are already set for local Ollama — nothing to edit

python scripts/make_sample_data.py    # generates data/messy_customers.csv
```

### Switching to hosted Claude

The agent is provider-agnostic (see `llm.py`). To swap backends, change two lines in `.env` — no code changes:

```bash
PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Local models are the default because the project should be runnable by anyone who clones it. Claude produces noticeably sharper findings — it's better at spotting subtle issues like target leakage and at writing pandas that works first try — so use it if you have a key.

### Run it — three ways

**1. CLI (fastest to demo)**
```bash
data-detective data/messy_customers.csv \
  --context "Customer churn export from our CRM. Target is churned. One row per customer." \
  --out report.md --fix-script cleanup.py
```

**2. API**
```bash
uvicorn data_detective.api:app --reload
# docs at http://127.0.0.1:8000/docs

curl -X POST http://127.0.0.1:8000/audits \
  -F "file=@data/messy_customers.csv" \
  -F "context=Churn data, target=churned"
# -> {"run_id":"a1b2c3d4e5f6","status":"running"}

curl http://127.0.0.1:8000/audits/a1b2c3d4e5f6
```

**3. Full UI** (two terminals)
```bash
uvicorn data_detective.api:app          # terminal 1
streamlit run src/data_detective/ui.py  # terminal 2 -> localhost:8501
```

### Verify the sandbox (no API key needed)

```bash
python scripts/demo_sandbox.py
```

Runs four real analysis queries and six real attack payloads against the sandbox directly:

```
LEGITIMATE ANALYSIS — should run
  RAN   | count placeholder ages     | 302
  RAN   | target leakage probe       | {0: 0.0, 1: 1.0}
  RAN   | exact duplicate rows       | 150

ATTACKS — should all be blocked
  BLOCK | read /etc/passwd           | use of 'open' is not allowed
  BLOCK | shell out via os           | import of 'os' is not allowed
  BLOCK | subclass sandbox escape    | '__subclasses__' is not allowed
  BLOCK | exfiltrate data to disk    | 'to_csv' may touch disk/network
  BLOCK | infinite loop (DoS)        | TIMEOUT after 5s

  Exfiltration file created?  False
  10/10 cases behaved as expected
```

The `{0: 0.0, 1: 1.0}` line is the target-leakage probe: `cancellation_reason` is populated for 100% of churned rows and 0% of retained ones. That column would leak the label straight into any model trained on it.

### Tests
```bash
pytest              # 40 tests, no API key or network required
pytest --cov=src
```
The agent tests use a fake Anthropic client. You cannot write reliable tests against a non-deterministic model, so the orchestration is tested with the model stubbed out — budgets, tool dispatch, error recovery, termination.

---

## The evaluation dataset

`scripts/make_sample_data.py` plants **nine known defects**, so you can measure whether the agent actually found things rather than generated plausible prose:

1. Placeholder poison — `age == 999` (300 rows)
2. Impossible values — `tenure_months == -1` (120 rows)
3. Real missingness — `monthly_charge` null (450 rows)
4. Inconsistent categories — `ny` / `N.Y.` / `New York` / `calif.`
5. Type mismatch — `last_payment` stored as `"$70.12"` strings
6. Zero variance — `data_source` has one value
7. **Target leakage** — `cancellation_reason` populated only when `churned == 1`
8. Duplicates — 150 exact duplicate rows
9. High cardinality — `notes` unique per row

Defect 7 is the interesting one. Nulls and duplicates are table stakes; catching that a feature is only ever populated for the positive class is the thing a junior analyst misses and a leaked model dies on.

---

## Security posture (read this before deploying)

The agent executes code it wrote itself. If a CSV contains a column named `__import__('os').system(...)`, naive `exec()` is remote code execution. Three independent layers:

1. **Static** — AST allowlist rejects imports, dunder access, `eval`/`exec`/`open`/`getattr`, and pandas' disk/network methods.
2. **Runtime** — separate `python -I` subprocess with stripped `__builtins__`, so a bypass lands in a crippled interpreter, not the app's.
3. **Resource** — `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE=0` (cannot write files at all), plus a wall-clock timeout.

Also: uploads are size-capped and streamed to disk, the user's filename is never used as a path, and the temp file is deleted after the run.

**This is not a true sandbox.** Escaping CPython restricted execution is a known research sport. For production, run the subprocess in a container with no network namespace and a read-only filesystem — the code is already isolated in `_runner.py` specifically so that swap is a deployment change, not a rewrite.

---

## Project structure

```
data-detective/
├── src/data_detective/
│   ├── config.py         # fail-fast settings from .env
│   ├── logging_setup.py  # structlog
│   ├── schemas.py        # Pydantic contracts + report rendering
│   ├── profiler.py       # deterministic stats — the agent's ground truth
│   ├── sandbox.py        # AST validation + isolated execution  ← security core
│   ├── _runner.py        # the child process (never imported by app code)
│   ├── tools.py          # tool schemas + dispatch + findings ledger
│   ├── memory.py         # transcript trimming
│   ├── agent.py          # the loop                              ← logic core
│   ├── api.py            # FastAPI
│   ├── ui.py             # Streamlit
│   └── cli.py            # CLI
├── scripts/make_sample_data.py
├── tests/
└── data/
```

---

## Known limitations

- Single-node storage; restarting the API loses in-flight runs.
- Loads the dataset into memory — capped at 500k rows by default. Larger data would need sampling or a DuckDB backend.
- Findings are LLM judgements. The evidence is real (it was measured), but severity is an opinion.
- No cross-run memory. Auditing the same table twice repeats the work.

---

## License

MIT
