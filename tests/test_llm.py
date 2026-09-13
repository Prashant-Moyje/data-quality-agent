"""Provider-layer tests.

The wire formats are the whole reason `llm.py` exists, and they are exactly the
kind of thing that breaks silently: a malformed transcript does not raise, it
just makes the model behave worse, or returns a 400 forty seconds into a run.

Nothing here touches the network or needs either SDK installed. The rendering
helpers are static methods, and `complete()` is exercised by building the
provider with `object.__new__` and handing it a stub client -- which is also how
you would test it if the SDK were an optional dependency.
"""

from __future__ import annotations

import pytest

from ground_truth.config import Settings
from ground_truth.llm import (
    AnthropicProvider,
    LLMError,
    OllamaProvider,
    ToolCall,
    build_provider,
)
from ground_truth.memory import ELIDED, Transcript

TOOLS = [
    {"name": "run_pandas", "description": "run code", "parameters": {"type": "object"}},
]


def _transcript() -> Transcript:
    """One full turn: user asks, model calls a tool, tool answers."""
    t = Transcript(keep_full_results=4)
    t.add_user("DATASET: x")
    t.add_assistant("", [ToolCall(id="c1", name="run_pandas", arguments={"code": "1"})])
    t.add_tool_results([{"id": "c1", "name": "run_pandas", "content": "42", "is_error": False}])
    return t


# ---------- Anthropic rendering ----------

def test_anthropic_tool_schema_uses_input_schema():
    out = AnthropicProvider._tools(TOOLS)
    assert out == [
        {"name": "run_pandas", "description": "run code", "input_schema": {"type": "object"}}
    ]


def test_anthropic_groups_tool_results_into_one_user_message():
    """The API rejects a tool_use with no matching tool_result in the SAME message."""
    t = Transcript()
    t.add_assistant("", [
        ToolCall(id="a", name="run_pandas", arguments={}),
        ToolCall(id="b", name="run_pandas", arguments={}),
    ])
    t.add_tool_results([
        {"id": "a", "name": "run_pandas", "content": "ok", "is_error": False},
        {"id": "b", "name": "run_pandas", "content": "boom", "is_error": True},
    ])

    msgs = AnthropicProvider._messages(t.for_api())

    results = [m for m in msgs if m["role"] == "user"]
    assert len(results) == 1, "both results must ride in one user message"
    blocks = results[0]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["a", "b"]
    assert blocks[0].get("is_error") is None  # success carries no flag
    assert blocks[1]["is_error"] is True


def test_anthropic_never_emits_an_empty_assistant_message():
    """A tool call with no prose still needs a non-empty content list."""
    t = Transcript()
    t.add_assistant("", [ToolCall(id="c1", name="run_pandas", arguments={})])
    msgs = AnthropicProvider._messages(t.for_api())
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"], "empty content is a 400"
    assert assistant["content"][0]["type"] == "tool_use"


def test_anthropic_keeps_pairing_after_trimming():
    """Trimming blanks content; it must never orphan a tool_use."""
    t = Transcript(keep_full_results=1)
    for i in range(4):
        t.add_assistant("", [ToolCall(id=f"c{i}", name="run_pandas", arguments={})])
        t.add_tool_results([{"id": f"c{i}", "name": "run_pandas", "content": "big", "is_error": False}])

    msgs = AnthropicProvider._messages(t.for_api())
    used = {b["id"] for m in msgs if m["role"] == "assistant"
            for b in m["content"] if b["type"] == "tool_use"}
    answered = {b["tool_use_id"] for m in msgs if m["role"] == "user"
                for b in m["content"] if b["type"] == "tool_result"}
    assert used == answered
    contents = [b["content"] for m in msgs if m["role"] == "user"
                for b in m["content"] if b["type"] == "tool_result"]
    assert contents.count(ELIDED) == 3 and contents[-1] == "big"


# ---------- Ollama rendering ----------

def test_ollama_tool_schema_is_openai_shaped():
    out = OllamaProvider._tools(TOOLS)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "run_pandas"
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_ollama_puts_system_first_and_splits_tool_results():
    """Ollama has no tool_use ids, so each result is its own message, named."""
    msgs = OllamaProvider._messages("SYS", _transcript().for_api())

    assert msgs[0] == {"role": "system", "content": "SYS"}
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith("[run_pandas]")
    assert "42" in tool_msgs[0]["content"]


# ---------- Ollama response parsing ----------

class _FakeOllamaClient:
    def __init__(self, message: dict, **usage):
        self._resp = {"message": message, **usage}

    def chat(self, **_kwargs):
        return self._resp


def _ollama_with(client) -> OllamaProvider:
    """Build the provider without importing the SDK."""
    p = object.__new__(OllamaProvider)
    p.client = client
    p.model = "qwen3:8b"
    p.options = {}
    return p


def test_ollama_mints_ids_and_reports_tokens():
    p = _ollama_with(_FakeOllamaClient(
        {"content": "thinking", "tool_calls": [
            {"function": {"name": "run_pandas", "arguments": {"code": "result = 1"}}}
        ]},
        prompt_eval_count=120, eval_count=30,
    ))

    resp = p.complete("sys", [], TOOLS)

    assert resp.text == "thinking"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "run_pandas"
    assert call.arguments == {"code": "result = 1"}
    assert call.id, "an id must be minted; the neutral transcript pairs on it"
    assert (resp.input_tokens, resp.output_tokens) == (120, 30)


def test_ollama_parses_arguments_delivered_as_a_json_string():
    """Some models emit arguments as a string rather than an object."""
    p = _ollama_with(_FakeOllamaClient(
        {"content": "", "tool_calls": [
            {"function": {"name": "run_pandas", "arguments": '{"code": "result = 2"}'}}
        ]},
    ))
    assert p.complete("sys", [], TOOLS).tool_calls[0].arguments == {"code": "result = 2"}


def test_ollama_survives_unparseable_arguments():
    """A small model emitting junk must not kill the run."""
    p = _ollama_with(_FakeOllamaClient(
        {"content": "", "tool_calls": [{"function": {"name": "run_pandas", "arguments": "{not json"}}]},
    ))
    resp = p.complete("sys", [], TOOLS)
    assert resp.tool_calls[0].arguments == {}  # the toolbox will reject it and say why


def test_ollama_handles_a_prose_only_reply():
    p = _ollama_with(_FakeOllamaClient({"content": "Let me think."}))
    resp = p.complete("sys", [], TOOLS)
    assert resp.tool_calls == []
    assert resp.text == "Let me think."


# ---------- the factory ----------

def test_anthropic_without_a_key_fails_fast_and_says_why():
    settings = Settings(provider="anthropic", anthropic_api_key="")  # type: ignore[call-arg]
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_provider(settings)


def test_unreachable_ollama_names_the_host_it_tried():
    """The failure a first-time user actually hits: `ollama serve` not running."""
    settings = Settings(provider="ollama", ollama_host="http://127.0.0.1:1")  # type: ignore[call-arg]
    pytest.importorskip("ollama", reason="ollama SDK not installed")
    with pytest.raises(LLMError, match="127.0.0.1:1"):
        build_provider(settings)
