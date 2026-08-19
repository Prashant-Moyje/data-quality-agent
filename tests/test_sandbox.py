"""Sandbox tests.

These are the tests that matter most. If the static checks regress, the app
executes arbitrary LLM-generated code with no guardrail. Each attack below is a
real escape technique, not a hypothetical.
"""

from __future__ import annotations

import pytest

from ground_truth.sandbox import UnsafeCodeError, run_snippet, validate_code


# ---------- Layer 1: static analysis ----------

SAFE_SNIPPETS = [
    "result = df.shape",
    "result = (df['age'] > 100).sum()",
    "import pandas as pd\nresult = pd.isna(df['monthly_charge']).mean()",
    "result = df.groupby('state').size().to_dict()",
    "print(df['plan'].value_counts())",
    "result = df[df.duplicated(subset=['customer_id'], keep=False)].shape[0]",
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
