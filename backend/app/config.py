"""
config.py – Central configuration loaded from environment variables.

Why a config module?
  Weights and model names change during experiments. Keeping them here (not
  scattered across files) means one edit updates the whole pipeline.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# If model is already cached locally, default to offline mode to avoid slow/hanging network checks
_cache_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
if _cache_path.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Settings(BaseSettings):
    """All tuneable parameters for the pipeline.

    Values are read from the .env file at startup.
    Pydantic-settings validates types automatically.
    """

    # Sentence-transformers model to use for embeddings
    model_name: str = "all-MiniLM-L6-v2"

    # Scoring fusion weights (must sum to 1.0 — validated below)
    semantic_weight: float = 0.6
    skill_weight: float = 0.4

    # LLM Feedback configuration (Phase 6)
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"

    # Uvicorn bind settings
    host: str = "127.0.0.1"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        total = round(self.semantic_weight + self.skill_weight, 10)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"semantic_weight + skill_weight must equal 1.0, got {total}"
            )


# Module-level singleton — import this everywhere
settings = Settings()
