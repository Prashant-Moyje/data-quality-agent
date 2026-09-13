from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Tests must never require a real key. Set a dummy before config is imported.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")
# ...and must not scatter run artefacts through the working tree. Settings reads
# this at import time, so it has to be set before anything imports config.
os.environ.setdefault(
    "STORAGE_DIR", str(Path(tempfile.gettempdir()) / "ground_truth_test_runs")
)


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "messy_customers.csv"
    if not path.exists():
        pytest.skip("run `python scripts/make_sample_data.py` first")
    return path
