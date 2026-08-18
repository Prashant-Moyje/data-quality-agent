"""Configuration, loaded once from environment / .env.

WHY pydantic-settings: it validates config at startup. A missing API key fails
immediately with a readable message instead of 40 seconds into an audit.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str = Field(..., description="From console.anthropic.com")

    # claude-sonnet-5 is the workhorse: strong tool-use + code generation at a
    # sane price. Swap to claude-opus-5 for harder reasoning, or
    # claude-haiku-4-5-20251001 to run the loop cheaply while developing.
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
