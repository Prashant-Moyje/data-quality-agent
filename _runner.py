"""Child process entry point for the sandbox. Never import this from app code.

Reads a JSON job from stdin, prints exactly one JSON line to stdout.
Runs with `python -I` (isolated) and, on POSIX, hard OS resource limits.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

MAX_CHARS = 4000

# A minimal builtins map. If a snippet somehow slips past the AST check, this is
# what it lands in: no open, no eval, no __import__.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}


def _apply_limits(memory_mb: int) -> None:
    """CPU, address space and file-size ceilings. POSIX only; no-op on Windows."""
    try:
        import resource
    except ImportError:  # Windows
        return
    soft_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (soft_bytes, soft_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # cannot write files at all
    # NOTE: deliberately NOT setting RLIMIT_NPROC=0 — on Linux threads count
    # toward it and numpy/OpenBLAS spawn worker threads.


def _stringify(value: object) -> str:
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return value.head(50).to_string()
    if isinstance(value, pd.Series):
        return value.head(50).to_string()
    return str(value)


def main() -> None:
    raw = sys.stdin.read()
    try:
        job = json.loads(raw)
        code: str = job["code"]
        data_path: str = job["data_path"]
        memory_mb: int = int(job.get("memory_mb", 1024))
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad job payload: {e}"}))
        return

    try:
        # Load the data BEFORE clamping memory, so a large legitimate file
        # doesn't get killed at read time.
        import numpy as np
        import pandas as pd

        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        elif data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path, low_memory=False)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"could not load data: {e}"}))
        return

    _apply_limits(memory_mb)

    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "result": None,
    }

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<snippet>", "exec"), namespace)  # noqa: S102
        out = {
            "ok": True,
            "stdout": buf.getvalue()[:MAX_CHARS],
            "result": _stringify(namespace.get("result"))[:MAX_CHARS]
            if namespace.get("result") is not None
            else "",
            "error": "",
        }
    except MemoryError:
        out = {"ok": False, "error": "MemoryError: query used too much memory."}
    except Exception as e:
        out = {
            "ok": False,
            "stdout": buf.getvalue()[:1000],
            "error": f"{type(e).__name__}: {e}",
        }

    sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
