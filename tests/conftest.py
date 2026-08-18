from __future__ import annotations

import os
from pathlib import Path

import pytest

# Tests must never require a real key. Set a dummy before config is imported.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "messy_customers.csv"
    if not path.exists():
        pytest.skip("run `python scripts/make_sample_data.py` first")
    return path
