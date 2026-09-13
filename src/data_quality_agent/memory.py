"""Working memory for the agent loop.

Stores a PROVIDER-NEUTRAL transcript. Each provider in `llm.py` renders this to
its own wire format, so trimming logic is written once instead of twice.

THE PROBLEM: chat APIs are stateless. Every turn resends the whole conversation,
so an agent that dumps DataFrames into the transcript grows quadratically in
cost and eventually overflows the context window. That hurts far more on a local
model, where the window is smaller and you pay for it in seconds, not cents.

THE FIX: elide the *content* of old tool results, keeping the last N in full.
The model rarely needs step 2's raw output once it has drawn a conclusion.
Never drop the result entry itself - Anthropic requires every tool_use to be
answered by a matching tool_result, and an unmatched call is a 400.

Durable state (findings) lives in ToolBox, not here, so trimming cannot
destroy it. Context and memory are different things.
"""

from __future__ import annotations

from typing import Any

from .llm import ToolCall

ELIDED = "[older tool output elided to save context - conclusions already recorded]"


class Transcript:
    def __init__(self, keep_full_results: int = 4):
        self.messages: list[dict[str, Any]] = []
        self.keep_full_results = keep_full_results

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "text": text})

    def add_assistant(self, text: str, tool_calls: list[ToolCall] | None = None) -> None:
        self.messages.append(
            {"role": "assistant", "text": text, "tool_calls": tool_calls or []}
        )

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """All results for one assistant turn, as {id, name, content, is_error}."""
        self.messages.append({"role": "tool_results", "results": results})

    def for_api(self) -> list[dict[str, Any]]:
        """A trimmed copy, safe to hand to a provider."""
        idxs = [i for i, m in enumerate(self.messages) if m["role"] == "tool_results"]
        to_elide = set(idxs[: -self.keep_full_results]) if self.keep_full_results else set()

        out: list[dict[str, Any]] = []
        for i, msg in enumerate(self.messages):
            if i not in to_elide:
                out.append(msg)
                continue
            out.append(
                {
                    "role": "tool_results",
                    "results": [{**r, "content": ELIDED} for r in msg["results"]],
                }
            )
        return out

    def __len__(self) -> int:
        return len(self.messages)
