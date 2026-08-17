"""Generate a deliberately messy dataset so the agent has something to find.

Every defect here is one a real CRM export actually produces. The point is a
reproducible demo where YOU know the ground truth, so you can judge whether the
agent actually found things or just produced plausible-sounding prose.

Ground truth planted (9 defects):
  1. Placeholder poison  — age == 999 for 300 rows
  2. Impossible values   — tenure_months == -1 for 120 rows
  3. Real missingness    — monthly_charge null for 450 rows
  4. Inconsistent cats   — state: 'ny' / 'N.Y.' / 'New York' / 'calif.'
  5. Type mismatch       — last_payment stored as '$70.12' strings
  6. Zero variance       — data_source has exactly one value
  7. Target leakage      — cancellation_reason only populated when churned==1
  8. Duplicates          — 150 exact duplicate rows (and duplicate customer_ids)
  9. High cardinality    — notes is unique per row
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

df = pd.DataFrame(
    {
        "customer_id": [f"C{i:05d}" for i in range(N)],
        "age": rng.integers(18, 85, N),
        "tenure_months": rng.integers(0, 120, N),
        "monthly_charge": np.round(rng.normal(70, 25, N), 2),
        "state": rng.choice(["CA", "NY", "TX", "FL", "WA"], N),
        "plan": rng.choice(["basic", "pro", "enterprise"], N, p=[0.6, 0.3, 0.1]),
        "signup_date": pd.to_datetime("2020-01-01")
        + pd.to_timedelta(rng.integers(0, 1400, N), "D"),
        "support_tickets": rng.poisson(1.2, N),
        "churned": rng.choice([0, 1], N, p=[0.88, 0.12]),
    }
)

df.loc[rng.choice(N, 300, replace=False), "age"] = 999
df.loc[rng.choice(N, 120, replace=False), "tenure_months"] = -1
df.loc[rng.choice(N, 450, replace=False), "monthly_charge"] = np.nan

mask = rng.choice(N, 400, replace=False)
df.loc[mask, "state"] = rng.choice(["ny", "N.Y.", "New York", "calif."], len(mask))

df["last_payment"] = [f"${abs(x):.2f}" for x in rng.normal(70, 20, N)]
df["data_source"] = "crm_export"
reasons = pd.Series(rng.choice(["price", "service", "moved"], N), index=df.index)
df["cancellation_reason"] = reasons.where(df["churned"] == 1, other=pd.NA)

# High-cardinality free text. NOTE: this must be added BEFORE duplicating rows,
# otherwise every row gets a unique note and the "exact duplicates" defect
# silently disappears. (The test suite caught exactly this.)
df["notes"] = [f"note-{i}" for i in range(len(df))]

dupes = df.sample(150, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

out = Path(__file__).resolve().parents[1] / "data" / "messy_customers.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out} — {len(df)} rows x {len(df.columns)} cols")
