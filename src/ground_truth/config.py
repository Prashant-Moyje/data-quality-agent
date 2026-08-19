"""Configuration, loaded once from environment / .env.

WHY pydantic-settings: it validates config at startup. A missing API key fails
immediately with a readable message instead of 40 seconds into an audit.
"""

from __future__ import annotations

from pathlib import Path

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Which brain? "ollama" runs entirely on your machine, no key, no cost.
    provider: Literal["ollama", "anthropic"] = "ollama"

    # -- Ollama (local) --
    # Tool calling is the hard requirement here. Models known to do it well:
    #   qwen3:8b        <- recommended default, best tool use per GB
    #   llama3.1:8b     <- solid fallback
    #   qwen2.5:14b     <- better if you have the VRAM
    # Small models (<7B) will produce malformed tool calls; see README.
    ollama_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"
    # Context window. Must be generous: the transcript is resent every turn and
    # Ollama's default is small enough that the system prompt falls out mid-run.
    ollama_num_ctx: int = 16384

    # -- Anthropic (hosted) --
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-5"

    max_tokens_per_call: int = 4096

    # --- Agent loop guardrails (the thing that separates a demo from a system) ---
    max_steps: int = 14           # hard cap on LLM turns
    max_tool_calls: int = 30      # hard cap on tool executions
    sandbox_timeout_s: int = 20   # per snippet
    sandbox_memory_mb: int = 1024

    # --- Limits on the data itself ---
    max_upload_mb: int = 50
    max_rows_scanned: int = 500_000

    # --- Cost tracking. Set from https://claude.com/pricing for your model. ---
    # Left at 0 by default so the app never reports a made-up dollar figure.
    price_per_mtok_input: float = 0.0
    price_per_mtok_output: float = 0.0

    log_level: str = "INFO"
    log_json: bool = False
    storage_dir: Path = Path("./runs")
    api_url: str = "http://127.0.0.1:8000"  # used by the Streamlit UI


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached accessor so we validate config exactly once per process."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
        _settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return _settings
