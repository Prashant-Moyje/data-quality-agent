"""Working memory for the agent loop.

THE PROBLEM: the Messages API is stateless. Every turn resends the whole
conversation, so a 14-step agent that dumps DataFrames into the transcript grows
quadratically in cost and eventually blows the context window.

THE FIX (two parts):
  1. Elide the *content* of old tool results, keeping the last N in full. The
     model rarely needs step 2's raw output once it's drawn a conclusion from it.
  2. Never delete a tool_result block. The API requires every `tool_use` to be
     answered by a `tool_result` with a matching id — deleting one is a 400.
     So we replace the text, not the block.

Durable state (findings) lives in ToolBox, not here, precisely so trimming
cannot destroy it.
"""

from __future__ import annotations

from typing import Any

ELIDED = "[older tool output elided to save context — conclusions already recorded]"


class Transcript:
    def __init__(self, keep_full_results: int = 4):
        self.messages: list[dict[str, Any]] = []
        self.keep_full_results = keep_full_results

    def add_user(self, content: str | list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """All tool results for one assistant turn go in ONE user message."""
        self.messages.append({"role": "user", "content": results})

    def for_api(self) -> list[dict[str, Any]]:
        """Return a trimmed copy safe to send to the API."""
        # Find indices of messages that carry tool_result blocks.
        tr_indices = [
            i
            for i, m in enumerate(self.messages)
            if isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
        ]
        to_elide = set(tr_indices[: -self.keep_full_results]) if self.keep_full_results else set()

        out: list[dict[str, Any]] = []
        for i, msg in enumerate(self.messages):
            if i not in to_elide:
                out.append(msg)
                continue
            blocks = []
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    # Keep the block + its id; replace only the payload.
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b["tool_use_id"],
                            "content": ELIDED,
                            **({"is_error": True} if b.get("is_error") else {}),
                        }
                    )
                else:
                    blocks.append(b)
            out.append({"role": msg["role"], "content": blocks})
        return out

    def __len__(self) -> int:
        return len(self.messages)
