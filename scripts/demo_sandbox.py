"""Demonstrate the sandbox: legitimate analysis runs, attacks are blocked.

Requires no API key and makes no network calls — this exercises the security
layer directly, not the agent.

    python scripts/demo_sandbox.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_detective.sandbox import run_snippet  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "data" / "messy_customers.csv"

LEGITIMATE = [
    ("count placeholder ages", "result = (df['age'] == 999).sum()"),
    ("inconsistent state labels", "result = df['state'].value_counts().to_dict()"),
    (
        "target leakage probe",
        "result = df.groupby('churned')['cancellation_reason']"
        ".apply(lambda s: s.notna().mean()).to_dict()",
    ),
    ("exact duplicate rows", "result = df.duplicated().sum()"),
]

ATTACKS = [
    ("read /etc/passwd", "result = open('/etc/passwd').read()"),
    ("shell out via os", "import os\nos.system('whoami')"),
    ("subclass sandbox escape", "result = ().__class__.__bases__[0].__subclasses__()"),
    ("exfiltrate data to disk", "df.to_csv('/tmp/stolen.csv')"),
    ("dynamic import", "result = __import__('os').getcwd()"),
    ("infinite loop (DoS)", "while True:\n    pass"),
]


def show(label: str, code: str, expect_ok: bool) -> bool:
    res = run_snippet(code, CSV, timeout_s=5)
    text = (res.result or res.stdout or res.error).replace("\n", " ")[:70]
    correct = res.ok is expect_ok
    verdict = "RAN  " if res.ok else "BLOCK"
    mark = "✓" if correct else "✗ UNEXPECTED"
    print(f"  {verdict} | {label:26} | {text:70} {mark}")
    return correct


def main() -> int:
    if not CSV.exists():
        print("Run `python scripts/make_sample_data.py` first.")
        return 2

    print("\nLEGITIMATE ANALYSIS — should run\n" + "-" * 110)
    ok = [show(l, c, expect_ok=True) for l, c in LEGITIMATE]

    print("\nATTACKS — should all be blocked\n" + "-" * 110)
    ok += [show(l, c, expect_ok=False) for l, c in ATTACKS]

    leaked = Path("/tmp/stolen.csv")
    print("\n" + "-" * 110)
    print(f"  Exfiltration file created?  {leaked.exists()}  (must be False)")
    print(f"  {sum(ok)}/{len(ok)} cases behaved as expected")
    return 0 if all(ok) and not leaked.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
