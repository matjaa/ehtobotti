"""Load settings from config.yaml into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class ChatModelOptions:
    think_supported: bool = False


@dataclass(frozen=True)
class ChunkingSettings:
    max_tokens: int = 400
    chars_per_token: int = 4
    max_heading_chars: int = 90

    @property
    def max_chars(self) -> int:
        return self.max_tokens * self.chars_per_token


@dataclass(frozen=True)
class Settings:
    provider: str
    ollama_host: str
    default_chat_model: str
    embedding_model: str
    chat_models: dict[str, ChatModelOptions]
    num_ctx: int
    temperature: float
    chunking: ChunkingSettings
    top_k: int
    embed_batch_size: int
    raw_dir: Path
    index_dir: Path
    sources_path: Path
    collection: str
    products: dict[str, str] = field(default_factory=dict)

    def chat_options(self, model: str) -> ChatModelOptions:
        """Options for a chat model; unknown models get safe defaults (no think flag)."""
        return self.chat_models.get(model, ChatModelOptions())


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    models = raw.get("models", {})
    paths = raw.get("paths", {})

    return Settings(
        provider=raw.get("provider", "ollama"),
        ollama_host=raw.get("ollama", {}).get("host", "http://localhost:11434"),
        default_chat_model=models["default_chat"],
        embedding_model=models["embedding"],
        chat_models={
            name: ChatModelOptions(**(opts or {}))
            for name, opts in (models.get("chat") or {}).items()
        },
        num_ctx=raw.get("generation", {}).get("num_ctx", 4096),
        temperature=raw.get("generation", {}).get("temperature", 0.1),
        chunking=ChunkingSettings(**raw.get("chunking", {})),
        top_k=raw.get("retrieval", {}).get("top_k", 5),
        embed_batch_size=raw.get("embedding", {}).get("batch_size", 32),
        raw_dir=_resolve(paths.get("raw_dir", "data/raw")),
        index_dir=_resolve(paths.get("index_dir", "data/index")),
        sources_path=_resolve(paths.get("sources", "data/sources.yaml")),
        collection=raw.get("index", {}).get("collection", "vakuutusehdot"),
        products=raw.get("products", {}),
    )
