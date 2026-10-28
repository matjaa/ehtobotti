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


def _where(**filters: str | None) -> dict | None:
    """Chroma metadata filter from the given non-empty fields."""
    clauses = [{key: value} for key, value in filters.items() if value]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


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
    def search(
        self, query: str, *, top_k: int | None = None, product: str | None = None, company: str | None = None
    ) -> list[RetrievedChunk]: ...


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

    def search(
        self, query: str, *, top_k: int | None = None, product: str | None = None, company: str | None = None
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=top_k or self._settings.top_k,
            where=_where(product=product, company=company),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(text=text, metadata=dict(meta), score=1.0 - distance)
            for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def get_section(self, section: str, page: int) -> list[RetrievedChunk]:
        """All indexed chunks of a section (or, for text before any section, of a page).

        Used to show what a cited section says when it was not among the retrieved chunks.
        """
        where = {"section": section} if section else {"$and": [{"section": ""}, {"page": page}]}
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        return [
            RetrievedChunk(text=text, metadata=dict(meta), score=0.0)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

    def has_product(self, product: str, company: str | None = None) -> bool:
        """True if at least one document of the product (and company) is indexed."""
        where = _where(product=product, company=company)
        return bool(self._collection.get(where=where, limit=1, include=[])["ids"])

    def companies(self, product: str | None = None) -> list[str]:
        """Companies with indexed documents, optionally for one product."""
        metadatas = self._collection.get(where=_where(product=product), include=["metadatas"])["metadatas"]
        return sorted({str(m.get("company", "")) for m in metadatas} - {""})
