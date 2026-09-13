"""Deterministic profiling.

DESIGN RULE: the LLM never computes a statistic.

Language models are bad at arithmetic and great at interpretation. So Python
computes every number, and the agent's job is to decide *what those numbers
mean* and *what to investigate next*. This one rule removes an entire class of
hallucination from the system.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pandas as pd

from .logging_setup import get_logger
from .schemas import ColumnProfile, DatasetProfile

log = get_logger(__name__)


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


def cache_for_sandbox(df: pd.DataFrame, run_id: str) -> Path | None:
    """Write the already-parsed frame once, for the sandbox to read.

    Every run_pandas call spawns a fresh interpreter that re-reads the dataset.
    The fresh interpreter is a security property, not an oversight -- but
    re-parsing the same CSV a dozen times in one audit is pure waste, and it also
    hid a correctness gap: the profiler stops at MAX_ROWS_SCANNED, so the sandbox
    was measuring rows the profile never described, and the agent was comparing
    the two as if they were the same dataset.

    Parquet preserves the dtypes pandas already inferred. Messy frames are
    exactly the ones that refuse to serialise (one object column holding two
    types), so CSV is the fallback, and None means "carry on with the original
    file" -- slower and uncapped, but never wrong about the data itself.
    """
    stem = Path(tempfile.gettempdir()) / f"dqa_{run_id}_{uuid.uuid4().hex[:6]}"

    parquet = stem.with_suffix(".parquet")
    try:
        df.to_parquet(parquet, index=False)
        return parquet
    except Exception as e:  # mixed dtypes, missing pyarrow, read-only tmp...
        log.info("sandbox_cache.parquet_failed", error=str(e)[:200])
        parquet.unlink(missing_ok=True)

    csv = stem.with_suffix(".csv")
    try:
        df.to_csv(csv, index=False)
        return csv
    except Exception as e:  # pragma: no cover - defensive
        log.warning("sandbox_cache.failed", error=str(e)[:200])
        csv.unlink(missing_ok=True)
        return None
