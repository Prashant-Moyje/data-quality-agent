"""Deterministic profiling.

DESIGN RULE: the LLM never computes a statistic.

Language models are bad at arithmetic and great at interpretation. So Python
computes every number, and the agent's job is to decide *what those numbers
mean* and *what to investigate next*. This one rule removes an entire class of
hallucination from the system.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import ColumnProfile, DatasetProfile


def load_dataframe(path: Path, max_rows: int) -> pd.DataFrame:
    """Load CSV/Parquet defensively.

    Note `dtype=str` is NOT used: we want pandas' inferred types, because a
    column pandas reads as `object` when it should be numeric is itself a
    finding worth reporting.
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        if len(df) > max_rows:
            df = df.head(max_rows)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=max_rows)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Use .csv, .parquet or .xlsx")

    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def _sample_values(series: pd.Series, k: int = 5) -> list[str]:
    vals = series.dropna().unique()[:k]
    out = []
    for v in vals:
        s = str(v)
        out.append(s if len(s) <= 40 else s[:37] + "...")
    return out


def profile_dataframe(df: pd.DataFrame, name: str) -> DatasetProfile:
    n_rows = len(df)
    columns: list[ColumnProfile] = []

    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())

        numeric_stats = None
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            d = s.dropna()
            numeric_stats = {
                "min": float(d.min()),
                "p25": float(d.quantile(0.25)),
                "p50": float(d.quantile(0.50)),
                "p75": float(d.quantile(0.75)),
                "max": float(d.max()),
                "mean": float(d.mean()),
                "std": float(d.std()) if len(d) > 1 else 0.0,
            }

        columns.append(
            ColumnProfile(
                name=str(col),
                dtype=str(s.dtype),
                null_count=null_count,
                null_pct=round(100 * null_count / n_rows, 2) if n_rows else 0.0,
                unique_count=int(s.nunique(dropna=True)),
                sample_values=_sample_values(s),
                numeric_stats=numeric_stats,
            )
        )

    return DatasetProfile(
        name=name,
        n_rows=n_rows,
        n_cols=len(df.columns),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        exact_duplicate_rows=int(df.duplicated().sum()),
        columns=columns,
    )
