"""Find the chunks most relevant to a question.

Phase 1 is vector search only. Phase 2 adds BM25 and reciprocal rank fusion behind
the same `search` signature, so the generation code does not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ehtobotti.config import Settings


class IndexNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity, higher is better

    @property
    def product(self) -> str:
        return str(self.metadata.get("product", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    @property
    def section_title(self) -> str:
        return str(self.metadata.get("section_title", ""))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))

    @property
    def document(self) -> str:
        return str(self.metadata.get("document", ""))


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int | None = None, product: str | None = None) -> list[RetrievedChunk]: ...


class VectorRetriever:
    def __init__(self, settings: Settings, embed: Callable[[list[str]], list[list[float]]]) -> None:
        import chromadb

        self._settings = settings
        self._embed = embed
        client = chromadb.PersistentClient(path=str(settings.index_dir))
        try:
            self._collection = client.get_collection(settings.collection)
        except Exception as exc:  # Chroma raises different types across versions
            raise IndexNotFoundError(settings.collection) from exc

    def search(self, query: str, *, top_k: int | None = None, product: str | None = None) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=top_k or self._settings.top_k,
            where={"product": product} if product else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(text=text, metadata=dict(meta), score=1.0 - distance)
            for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
