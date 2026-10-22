"""LLM provider abstraction.

The rest of the code talks to `LLMClient` only, so moving from local Ollama to a
cloud model (e.g. Azure OpenAI in phase 3) is a config change, not a rewrite.
"""

from __future__ import annotations

from typing import Any, Protocol

from ehtobotti.config import Settings


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, **opts: Any) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        import ollama  # imported lazily so tests run without the package installed

        self._settings = settings
        self._client = ollama.Client(host=settings.ollama_host)

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, **opts: Any) -> str:
        model = model or self._settings.default_chat_model
        options = {
            "num_ctx": self._settings.num_ctx,
            "temperature": self._settings.temperature,
            **opts,
        }
        kwargs: dict[str, Any] = {}
        # Only thinking-capable models accept the flag; Gemma rejects it.
        if self._settings.chat_options(model).think_supported:
            kwargs["think"] = False
        response = self._client.chat(model=model, messages=messages, options=options, **kwargs)
        return response.message.content

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self._settings.embedding_model, input=texts)
        return [list(vector) for vector in response.embeddings]


class AzureOpenAIClient:
    """Documented stub for phase 3. Not implemented on purpose: no cloud dependencies yet."""

    def __init__(self, settings: Settings) -> None:
        raise NotImplementedError("Azure OpenAI provider is planned for phase 3.")

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, **opts: Any) -> str:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def get_client(settings: Settings) -> LLMClient:
    providers = {"ollama": OllamaClient, "azure_openai": AzureOpenAIClient}
    try:
        return providers[settings.provider](settings)
    except KeyError:
        raise ValueError(f"Unknown provider in config.yaml: {settings.provider!r}") from None
