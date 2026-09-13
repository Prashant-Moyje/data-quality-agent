"""Sandbox tests.

These are the tests that matter most. If the static checks regress, the app
executes arbitrary LLM-generated code with no guardrail. Each attack below is a
real escape technique, not a hypothetical.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_quality_agent import sandbox
from data_quality_agent.sandbox import UnsafeCodeError, run_snippet, validate_code


# ---------- Layer 1: static analysis ----------

SAFE_SNIPPETS = [
    "result = df.shape",
    "result = (df['age'] > 100).sum()",
    "import pandas as pd\nresult = pd.isna(df['monthly_charge']).mean()",
    "result = df.groupby('state').size().to_dict()",
    "print(df['plan'].value_counts())",
    "result = df[df.duplicated(subset=['customer_id'], keep=False)].shape[0]",
    # Regression: the module-attribute denylist must not eat everyday pandas.
    "result = df.dtypes.to_dict()",
    "result = df['state'].value_counts().to_dict()",
    "result = pd.api.types.is_numeric_dtype(df['age'])",
]

ATTACKS = [
    ("os import",            "import os\nos.system('id')"),
    ("subprocess",           "from subprocess import run\nrun(['ls'])"),
    ("builtins escape",      "result = ().__class__.__bases__[0].__subclasses__()"),
    ("eval",                 "result = eval('1+1')"),
    ("exec",                 "exec('x=1')"),
    ("open file",            "result = open('/etc/passwd').read()"),
    ("dunder import",        "result = __import__('os').getcwd()"),
    ("getattr indirection",  "result = getattr(df, 'to_csv')('/tmp/x.csv')"),
    ("write via pandas",     "df.to_csv('/tmp/leak.csv')"),
    ("read another file",    "result = pd.read_csv('/etc/hosts')"),
    ("globals access",       "result = globals()"),
    ("dunder attr",          "result = df.__class__.__module__"),
]


@pytest.mark.parametrize("code", SAFE_SNIPPETS)
def test_legitimate_analysis_is_allowed(code: str) -> None:
    validate_code(code)  # must not raise


@pytest.mark.parametrize("name,code", ATTACKS, ids=[a[0] for a in ATTACKS])
def test_attacks_are_blocked(name: str, code: str) -> None:
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


def test_syntax_error_is_reported_not_raised_as_crash() -> None:
    with pytest.raises(UnsafeCodeError, match="SyntaxError"):
        validate_code("result = df[")


# ---------- Layer 1 + 2: module traversal ----------
# pandas and numpy both import `os` at module level, and `pd` / `np` are in the
# snippet namespace by design. So `pd.io.common.os.system(...)` reaches the real
# os module using no import statement, no dunder and no forbidden name. This was
# a live escape: it ran, and returned the process working directory.
#
# Closed in two places, and both are tested, because the whole point of layered
# defence is that each layer holds on its own:
#   layer 1  validate_code rejects the known module attribute names, so the
#            model gets an early, readable error it can rewrite against
#   layer 2  _runner.GuardedModule resolves the attribute and refuses anything
#            that IS a module, which catches paths nobody enumerated

MODULE_TRAVERSAL_ATTACKS = [
    ("os via pandas.io",      "result = pd.io.common.os.getcwd()"),
    ("os via pandas.compat",  "result = pd.compat.os.environ"),
    ("os via pandas.util",    "result = pd.util._print_versions.os.getcwd()"),
    ("shell via pandas.io",   "pd.io.common.os.system('id')"),
    ("pickle exec via numpy", "result = np.load('payload.npy', allow_pickle=True)"),
    ("pandas eval engine",    "result = df.eval('age + 1')"),
    ("pandas query engine",   "result = df.query('age > 900')"),
]


@pytest.mark.parametrize(
    "name,code", MODULE_TRAVERSAL_ATTACKS, ids=[a[0] for a in MODULE_TRAVERSAL_ATTACKS]
)
def test_module_traversal_blocked_statically(name: str, code: str) -> None:
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


def _run_unvalidated(code: str, csv: Path) -> dict:
    """Run a snippet in the child process, deliberately SKIPPING validate_code.

    Layer 1 now rejects these before they get here, so bypassing it is the only
    way to show layer 2 stops them by itself. If someone later loosens the AST
    rules, these tests keep failing loudly.
    """
    runner = Path(sandbox.__file__).parent / "_runner.py"
    payload = json.dumps({"code": code, "data_path": str(csv), "memory_mb": 512})
    proc = subprocess.run(
        [sys.executable, "-I", str(runner)],
        input=payload, capture_output=True, text=True, timeout=60,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


RUNTIME_TRAVERSAL_ATTACKS = [a for a in MODULE_TRAVERSAL_ATTACKS if "os" in a[0] or "shell" in a[0]]


@pytest.mark.parametrize(
    "name,code", RUNTIME_TRAVERSAL_ATTACKS, ids=[a[0] for a in RUNTIME_TRAVERSAL_ATTACKS]
)
def test_module_traversal_blocked_at_runtime(name: str, code: str, sample_csv: Path) -> None:
    out = _run_unvalidated(code, sample_csv)
    assert out["ok"] is False
    assert "module traversal blocked" in out["error"]


def test_guarded_module_still_allows_real_analysis(sample_csv: Path) -> None:
    """The guard must not cost the agent the dtype predicates it actually uses."""
    out = _run_unvalidated("result = pd.api.types.is_numeric_dtype(df['age'])", sample_csv)
    assert out["ok"] is True, out.get("error")
    assert out["result"].strip() == "True"

    out = _run_unvalidated("result = float(pd.isna(df['monthly_charge']).mean())", sample_csv)
    assert out["ok"] is True, out.get("error")


def test_guard_fails_closed_on_an_unlisted_submodule(sample_csv: Path) -> None:
    """`pd.api` is walkable, but only as far as the one path that was allowed."""
    out = _run_unvalidated("result = pd.api.executors", sample_csv)
    assert out["ok"] is False
    assert "module traversal blocked" in out["error"]


# ---------- Layer 2/3: actual execution ----------

def test_executes_and_returns_result(sample_csv):
    res = run_snippet("result = df.shape[0]", sample_csv)
    assert res.ok
    assert "5150" in res.result


def test_captures_stdout(sample_csv):
    res = run_snippet("print('hello from sandbox')", sample_csv)
    assert res.ok
    assert "hello from sandbox" in res.stdout


def test_runtime_error_is_returned_not_raised(sample_csv):
    res = run_snippet("result = df['column_that_does_not_exist']", sample_csv)
    assert res.ok is False
    assert "KeyError" in res.error
    # The agent must be able to read this and rewrite the query.
    assert "column_that_does_not_exist" in res.error


def test_blocked_code_returns_failure_not_exception(sample_csv):
    res = run_snippet("import os", sample_csv)
    assert res.ok is False
    assert "BLOCKED" in res.error


def test_infinite_loop_times_out(sample_csv):
    res = run_snippet("while True:\n    pass", sample_csv, timeout_s=3)
    assert res.ok is False
    assert "TIMEOUT" in res.error


def test_output_is_truncated(sample_csv):
    res = run_snippet("result = df", sample_csv)
    assert res.ok
    assert len(res.result) <= 4000  # a 5150-row frame must not flood the context


def test_leakage_probe_still_runs_end_to_end(sample_csv):
    """The query that catches the planted leakage defect must not be collateral."""
    res = run_snippet(
        "result = df.groupby('churned')['cancellation_reason']"
        ".apply(lambda s: s.notna().mean()).to_dict()",
        sample_csv,
    )
    assert res.ok, res.error
    assert "1.0" in res.result
