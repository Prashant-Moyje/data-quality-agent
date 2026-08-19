"""LLM provider layer.

WHY THIS EXISTS
---------------
Anthropic and Ollama disagree about almost everything at the wire level:

                    Anthropic                      Ollama (OpenAI-style)
  tool schema       {name, input_schema}           {type: function, function: {...}}
  tool call         content block, has an id       message.tool_calls, NO id
  tool result       user msg w/ tool_result block  separate {role: "tool"} message
  system prompt     top-level `system` param       a {role: "system"} message
  token counts      usage.input_tokens             prompt_eval_count

So `agent.py` speaks a neutral dialect (below) and each provider renders it to
its own format. The agent loop never learns which backend it's talking to,
which is why swapping them is a one-line env change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .logging_setup import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Neutral types
# --------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    """What agent.py depends on. Nothing else."""

    name: str
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Provider failed in a way the agent cannot recover from."""


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 4096):
        import anthropic  # imported lazily so Ollama users needn't install it

        self._sdk = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})

            elif m["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})

            elif m["role"] == "tool_results":
                # Anthropic requires ALL results for one turn in ONE user message.
                blocks = []
                for r in m["results"]:
                    b: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": r["id"],
                        "content": r["content"],
                    }
                    if r.get("is_error"):
                        b["is_error"] = True
                    blocks.append(b)
                out.append({"role": "user", "content": blocks})
        return out

    def complete(self, system, messages, tools) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    # Cached because the system prompt is resent every turn.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self._tools(tools),
            messages=self._messages(messages),
        )
        return LLMResponse(
            text=" ".join(b.text for b in resp.content if b.type == "text"),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
                for b in resp.content
                if b.type == "tool_use"
            ],
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )

    def retryable(self) -> tuple[type[Exception], ...]:
        return (
            self._sdk.RateLimitError,
            self._sdk.APIConnectionError,
            self._sdk.InternalServerError,
        )


# --------------------------------------------------------------------------
# Ollama (local)
# --------------------------------------------------------------------------
class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 16384,
        num_predict: int = 2048,
        temperature: float = 0.1,
    ):
        import ollama

        self._sdk = ollama
        self.client = ollama.Client(host=host)
        self.model = model
        # num_ctx matters enormously here. Ollama's default context is small, and
        # this agent resends a growing transcript every turn — too small a window
        # and the model silently forgets the system prompt mid-audit.
        self.options = {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        }

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})

            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("text", "")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)

            elif m["role"] == "tool_results":
                # Ollama wants ONE message per result, and has no id to match on,
                # so we name the tool in the content to keep it unambiguous.
                for r in m["results"]:
                    out.append(
                        {
                            "role": "tool",
                            "content": f"[{r['name']}] {r['content']}",
                        }
                    )
        return out

    def complete(self, system, messages, tools) -> LLMResponse:
        resp = self.client.chat(
            model=self.model,
            messages=self._messages(system, messages),
            tools=self._tools(tools),
            options=self.options,
        )
        msg = resp["message"]

        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models emit a JSON string
                import json

                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    log.warning("ollama.bad_arguments", raw=args[:200])
                    args = {}
            calls.append(
                # Ollama supplies no call id, so we mint one to keep the neutral
                # transcript uniform.
                ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=fn["name"], arguments=dict(args))
            )

        return LLMResponse(
            text=msg.get("content", "") or "",
            tool_calls=calls,
            input_tokens=resp.get("prompt_eval_count", 0) or 0,
            output_tokens=resp.get("eval_count", 0) or 0,
        )

    def retryable(self) -> tuple[type[Exception], ...]:
        import httpx

        return (httpx.ConnectError, httpx.ReadTimeout, self._sdk.ResponseError)


# --------------------------------------------------------------------------
def build_provider(settings) -> LLMProvider:
    """Factory. The only place that knows both backends exist."""
    if settings.provider == "ollama":
        try:
            p = OllamaProvider(
                model=settings.ollama_model,
                host=settings.ollama_host,
                num_ctx=settings.ollama_num_ctx,
                num_predict=settings.max_tokens_per_call,
            )
        except ImportError as e:
            raise LLMError("Ollama selected but `pip install ollama` is missing.") from e

        # Fail fast with an actionable message rather than mid-audit.
        try:
            available = [m["model"] for m in p.client.list()["models"]]
        except Exception as e:
            raise LLMError(
                f"Cannot reach Ollama at {settings.ollama_host}. Is `ollama serve` running?"
            ) from e
        if not any(m.startswith(settings.ollama_model.split(":")[0]) for m in available):
            raise LLMError(
                f"Model {settings.ollama_model!r} not found. Run: "
                f"ollama pull {settings.ollama_model}\nAvailable: {available or 'none'}"
            )
        log.info("provider.ollama", model=settings.ollama_model, host=settings.ollama_host)
        return p

    if not settings.anthropic_api_key:
        raise LLMError("PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
    log.info("provider.anthropic", model=settings.model)
    return AnthropicProvider(
        settings.anthropic_api_key, settings.model, settings.max_tokens_per_call
    )
