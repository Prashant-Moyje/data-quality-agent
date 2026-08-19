# Ã°Å¸â€Å½ Data Detective

**An autonomous agent that audits a dataset the way a senior data scientist would: it forms hypotheses about what's broken, writes its own pandas code to test them, runs that code in a sandbox, and reports only what it can prove with numbers.**

You give it a CSV. It gives you a severity-ranked list of data quality problems, each backed by evidence it actually measured Ã¢â‚¬â€ plus a runnable cleaning script.

```bash
data-detective data/messy_customers.csv --context "Customer churn export. Target column is churned."
```

---

## Engineering decisions

| Problem | Solution in this repo |
|---|---|
| LLMs are bad at arithmetic | `profiler.py` computes every statistic in Python. The model never does math, only interpretation. |
| LLM-generated code is remote code execution | `sandbox.py` Ã¢â‚¬â€ AST allowlist + isolated subprocess + OS resource limits. |
| Models assert things they didn't verify | Pydantic validator rejects any finding whose evidence contains no numbers. |
| Agents loop forever / burn money | Hard step, tool-call, timeout and memory budgets, with a "final step" warning injected so the model exits gracefully. |
| Stateless API = quadratic token cost | `memory.py` elides old tool output while preserving `tool_use`/`tool_result` pairing. |
| Findings must survive context trimming | Findings live in `ToolBox`, not the transcript. Context Ã¢â€°Â  memory. |

---

## Architecture

```
                    Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
   CSV upload  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ Ã¢â€â€š  profiler.py Ã¢â€â€š  deterministic facts (no LLM)
                    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                           Ã¢â€“Â¼
        Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
        Ã¢â€â€š            agent.py loop             Ã¢â€â€š
        Ã¢â€â€š                                      Ã¢â€â€š
        Ã¢â€â€š   Claude Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ picks a tool Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â        Ã¢â€â€š
        Ã¢â€â€š      Ã¢â€“Â²                      Ã¢â€“Â¼        Ã¢â€â€š
        Ã¢â€â€š      Ã¢â€â€š              Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â Ã¢â€â€š
        Ã¢â€â€š      Ã¢â€â€š              Ã¢â€â€š  run_pandas  Ã¢â€â€šÃ¢â€â‚¬Ã¢â€Â¼Ã¢â€â‚¬Ã¢â€“Â¶ sandbox.py Ã¢â€â‚¬Ã¢â€“Â¶ subprocess
        Ã¢â€â€š      Ã¢â€â€š              Ã¢â€â€šrecord_findingÃ¢â€â€šÃ¢â€â‚¬Ã¢â€Â¼Ã¢â€â‚¬Ã¢â€“Â¶ Pydantic validation
        Ã¢â€â€š      Ã¢â€â€š              Ã¢â€â€š finish_audit Ã¢â€â€š Ã¢â€â€š
        Ã¢â€â€š      Ã¢â€â€š              Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ Ã¢â€â€š
        Ã¢â€â€š      Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ tool result Ã¢â€”â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ         Ã¢â€â€š
        Ã¢â€â€š           (budget-capped)            Ã¢â€â€š
        Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                           Ã¢â€“Â¼
              AuditReport  Ã¢â€ â€™  markdown report + fix_script.py
```

**Flow:** profile deterministically Ã¢â€ â€™ agent hypothesises Ã¢â€ â€™ sandboxed execution Ã¢â€ â€™ validated findings Ã¢â€ â€™ structured report.

---

## Tech choices, and why

| Choice | Why | What I rejected |
|---|---|---|
| **Claude Sonnet 5** via the official `anthropic` SDK | Strong at tool use and code generation, which is the entire job here. Native prompt caching cuts cost on the resent system prompt. | Ã¢â‚¬â€ |
| **No agent framework** | The loop is ~60 lines and is the intellectual content of the project. A framework would hide it behind an abstraction I'd then have to debug. | LangChain, LangGraph, CrewAI. LangGraph would be the right call *if* this needed resumable/durable runs or human approval gates. It doesn't. |
| **Pydantic v2** | Turns "the model returned something weird" from a silent corruption into a typed error I can feed back to the model so it self-corrects. | Parsing JSON by hand |
| **Subprocess + AST allowlist** | Defence in depth against prompt injection hidden inside the data itself. | `exec()` in-process (RCE), full Docker-per-call (too slow for a demo) |
| **FastAPI + background tasks** | Audits take 30Ã¢â‚¬â€œ90s. Async job + polling is the standard shape for agent-backed APIs; a sync endpoint would hit proxy timeouts. | Sync endpoint, Celery (infrastructure with nothing to prove) |
| **Streamlit as a thin API client** | Holds zero agent logic, which proves the backend is genuinely reusable. | Streamlit calling the agent directly |
| **structlog** | Agent runs are non-deterministic and multi-step. You need to grep by `run_id` and step. | `print()`, stdlib f-string logging |
| **In-memory dict + JSON files** | Single-node demo. Swapping in Postgres is a repo change, not a redesign. | Postgres/Redis for a portfolio project |

---

## Setup

**Requires Python 3.10+ and [Ollama](https://ollama.com/download). No API key, no account, no cost - the model runs on your machine.**

```bash
git clone https://github.com/Prashant-Moyje/data-detective.git
cd data-detective

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

ollama pull qwen3:8b                  # ~5 GB, one time

cp .env.example .env                  # Windows cmd: copy .env.example .env
# defaults are already set for local Ollama - nothing to edit

python scripts/make_sample_data.py    # generates data/messy_customers.csv
```

### Run it Ã¢â‚¬â€ three ways

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

### Tests
```bash
pytest              # 40 tests, no API key or network required
pytest --cov=src
```
The agent tests use a fake Anthropic client. You cannot write reliable tests against a non-deterministic model, so the orchestration is tested with the model stubbed out Ã¢â‚¬â€ budgets, tool dispatch, error recovery, termination.

---

## The evaluation dataset

`scripts/make_sample_data.py` plants **nine known defects**, so you can measure whether the agent actually found things rather than generated plausible prose:

1. Placeholder poison Ã¢â‚¬â€ `age == 999` (300 rows)
2. Impossible values Ã¢â‚¬â€ `tenure_months == -1` (120 rows)
3. Real missingness Ã¢â‚¬â€ `monthly_charge` null (450 rows)
4. Inconsistent categories Ã¢â‚¬â€ `ny` / `N.Y.` / `New York` / `calif.`
5. Type mismatch Ã¢â‚¬â€ `last_payment` stored as `"$70.12"` strings
6. Zero variance Ã¢â‚¬â€ `data_source` has one value
7. **Target leakage** Ã¢â‚¬â€ `cancellation_reason` populated only when `churned == 1`
8. Duplicates Ã¢â‚¬â€ 150 exact duplicate rows
9. High cardinality Ã¢â‚¬â€ `notes` unique per row

Defect 7 is the interesting one. Nulls and duplicates are table stakes; catching that a feature is only ever populated for the positive class is the thing a junior analyst misses and a leaked model dies on.

---

## Security posture (read this before deploying)

The agent executes code it wrote itself. If a CSV contains a column named `__import__('os').system(...)`, naive `exec()` is remote code execution. Three independent layers:

1. **Static** Ã¢â‚¬â€ AST allowlist rejects imports, dunder access, `eval`/`exec`/`open`/`getattr`, and pandas' disk/network methods.
2. **Runtime** Ã¢â‚¬â€ separate `python -I` subprocess with stripped `__builtins__`, so a bypass lands in a crippled interpreter, not the app's.
3. **Resource** Ã¢â‚¬â€ `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE=0` (cannot write files at all), plus a wall-clock timeout.

Also: uploads are size-capped and streamed to disk, the user's filename is never used as a path, and the temp file is deleted after the run.

**This is not a true sandbox.** Escaping CPython restricted execution is a known research sport. For production, run the subprocess in a container with no network namespace and a read-only filesystem Ã¢â‚¬â€ the code is already isolated in `_runner.py` specifically so that swap is a deployment change, not a rewrite.

---

## Project structure

```
data-detective/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ src/data_detective/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config.py         # fail-fast settings from .env
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ logging_setup.py  # structlog
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ schemas.py        # Pydantic contracts + report rendering
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ profiler.py       # deterministic stats Ã¢â‚¬â€ the agent's ground truth
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ sandbox.py        # AST validation + isolated execution  Ã¢â€ Â security core
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ _runner.py        # the child process (never imported by app code)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tools.py          # tool schemas + dispatch + findings ledger
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ memory.py         # transcript trimming
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ agent.py          # the loop                              Ã¢â€ Â logic core
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ api.py            # FastAPI
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ ui.py             # Streamlit
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ cli.py            # CLI
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ scripts/make_sample_data.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tests/
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ data/
```

---

## Known limitations

- Single-node storage; restarting the API loses in-flight runs.
- Loads the dataset into memory Ã¢â‚¬â€ capped at 500k rows by default. Larger data would need sampling or a DuckDB backend.
- Findings are LLM judgements. The evidence is real (it was measured), but severity is an opinion.
- No cross-run memory. Auditing the same table twice repeats the work.

---

## License

MIT
