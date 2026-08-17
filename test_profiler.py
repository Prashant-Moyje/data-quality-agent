"""The profiler is the agent's ground truth. If it lies, everything downstream lies."""

from __future__ import annotations

import pandas as pd
import pytest

from data_detective.profiler import load_dataframe, profile_dataframe


def test_profile_matches_pandas(sample_csv):
    df = pd.read_csv(sample_csv)
    p = profile_dataframe(df, "messy_customers.csv")

    assert p.n_rows == len(df)
    assert p.n_cols == len(df.columns)
    assert p.exact_duplicate_rows == int(df.duplicated().sum())
    assert p.exact_duplicate_rows > 0  # we planted 150

    charge = next(c for c in p.columns if c.name == "monthly_charge")
    assert charge.null_count == int(df["monthly_charge"].isna().sum())
    assert charge.numeric_stats is not None

    # A text column must not get numeric stats.
    payment = next(c for c in p.columns if c.name == "last_payment")
    assert payment.numeric_stats is None


def test_zero_variance_column_is_visible_to_agent(sample_csv):
    df = pd.read_csv(sample_csv)
    p = profile_dataframe(df, "x")
    source = next(c for c in p.columns if c.name == "data_source")
    assert source.unique_count == 1  # the agent should catch this from the profile alone


def test_prompt_text_is_compact_and_complete(sample_csv):
    df = pd.read_csv(sample_csv)
    text = profile_dataframe(df, "x").to_prompt_text()
    for col in df.columns:
        assert col in text
    assert len(text) < 8000  # must not eat the context window


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")
    with pytest.raises(ValueError, match="empty"):
        load_dataframe(p, max_rows=100)


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataframe(p, max_rows=100)
