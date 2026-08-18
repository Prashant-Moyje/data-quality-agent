"""Command-line entry point: `python -m data_detective.cli data/messy_customers.csv`

Useful for CI, for demoing without two servers running, and for debugging the
agent loop with full logs visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import AuditAgent
from .config import get_settings
from .logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a dataset for quality problems.")
    parser.add_argument("dataset", type=Path, help="Path to .csv / .parquet / .xlsx")
    parser.add_argument("--context", default="", help="What this data is, and what the target column is.")
    parser.add_argument("--out", type=Path, default=None, help="Write the markdown report here.")
    parser.add_argument("--fix-script", type=Path, default=None, help="Write the cleaning script here.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"error: {args.dataset} not found", file=sys.stderr)
        return 2

    settings = get_settings()
    setup_logging("DEBUG" if args.verbose else settings.log_level, settings.log_json)

    report = AuditAgent(settings).audit(
        args.dataset,
        user_context=args.context,
        on_progress=lambda m: print(f"  · {m}", file=sys.stderr),
    )

    if report.status == "failed":
        print(f"AUDIT FAILED: {report.error}", file=sys.stderr)
        return 1

    md = report.to_markdown()
    print(md)
    if args.out:
        args.out.write_text(md)
        print(f"\n[wrote {args.out}]", file=sys.stderr)
    if args.fix_script:
        args.fix_script.write_text(report.to_fix_script())
        print(f"[wrote {args.fix_script}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
