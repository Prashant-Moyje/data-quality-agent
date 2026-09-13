"""Child process entry point for the sandbox. Never import this from app code.

Reads a JSON job from stdin, prints exactly one JSON line to stdout.
Runs with `python -I` (isolated) and, on POSIX, hard OS resource limits.
"""

from __future__ import annotations

import io
import json
import sys
import types
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


class SandboxViolation(Exception):
    """A snippet reached for something the sandbox does not expose."""


# Attribute names that are an escape hatch regardless of what object they hang
# off: pandas' expression engines call eval() internally, and numpy's disk
# entry points unpickle, which is arbitrary code execution by itself.
DENIED_ATTRS = frozenset({
    "eval", "query",
    "load", "save", "savez", "savez_compressed", "memmap",
    "fromfile", "tofile", "genfromtxt", "loadtxt", "savetxt",
})

# Module paths a snippet may legitimately walk into. `pd.api.types` holds the
# dtype predicates (is_numeric_dtype and friends) and exposes no modules of its
# own, so allowing it costs nothing. Everything else fails closed.
ALLOWED_SUBMODULES = {
    "pd": frozenset({"api", "api.types"}),
    "np": frozenset(),
}


class GuardedModule:
    """A module view that refuses to hand out other modules.

    This is the layer that closes module traversal. The AST allowlist works on
    names and cannot see types, so it has no way to know that `pd.io.common.os`
    terminates at the real `os` module -- no import statement, no dunder, no
    forbidden name anywhere in that expression. This wrapper resolves the
    attribute and then looks at what came back: if it is a module, it is refused
    unless its dotted path was explicitly allowed. Unknown paths fail closed, so
    a pandas upgrade that adds a new submodule does not silently open a hole.
    """

    __slots__ = ("_mod", "_root", "_path")

    def __init__(self, module: types.ModuleType, root: str, path: str = "") -> None:
        object.__setattr__(self, "_mod", module)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_path", path)

    def _label(self, name: str) -> str:
        return ".".join(x for x in (self._root, self._path, name) if x)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise SandboxViolation(f"{self._label(name)} is not available in the sandbox")
        if name in DENIED_ATTRS:
            raise SandboxViolation(
                f"{self._label(name)} is not allowed (it can execute code or touch disk)"
            )

        value = getattr(self._mod, name)

        if isinstance(value, types.ModuleType):
            path = f"{self._path}.{name}".lstrip(".") if self._path else name
            if path in ALLOWED_SUBMODULES.get(self._root, frozenset()):
                return GuardedModule(value, self._root, path)
            raise SandboxViolation(
                f"module traversal blocked: {self._label(name)} is a module. "
                f"Only the {self._root} functions and types are exposed -- "
                f"work through `df`, or use {self._root} helpers directly."
            )
        return value

    def __setattr__(self, name: str, value: object) -> None:
        raise SandboxViolation(f"cannot assign to {self._label(name)}")

    def __dir__(self) -> list[str]:
        return [n for n in dir(self._mod) if not n.startswith("_")]

    def __repr__(self) -> str:
        return f"<guarded module {self._root}>"


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
        # Guarded views, not the real modules: see GuardedModule above. The
        # DataFrame itself stays real -- it is data, and every module reachable
        # from it goes through a dunder the AST layer already rejects.
        "pd": GuardedModule(pd, "pd"),
        "np": GuardedModule(np, "np"),
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
    except SandboxViolation as e:
        # Reported like any other failure so the agent reads it and rewrites,
        # rather than the run dying on a blocked query.
        out = {"ok": False, "stdout": buf.getvalue()[:1000], "error": f"BLOCKED: {e}"}
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
