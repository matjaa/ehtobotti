"""Build the vector index: PDF -> text -> section chunks -> embeddings -> Chroma."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from ehtobotti.chunking import Chunk, chunk_document
from ehtobotti.config import Settings
from ehtobotti.pdftext import read_pdf


@dataclass(frozen=True)
class SourceInfo:
    product: str
    file: str
    title: str
    version: str
    url: str = ""
    label: str = ""


def load_sources(path: Path) -> dict[tuple[str, str], SourceInfo]:
    """Map (product, filename) to the document's public metadata."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    return {
        (e["product"], e["file"]): SourceInfo(
            product=e["product"],
            file=e["file"],
            title=e.get("title", Path(e["file"]).stem),
            version=str(e.get("version", "")),
            url=e.get("url", ""),
            label=e.get("label", ""),
        )
        for e in entries
    }


def _doc_id(product: str, pdf: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", pdf.stem.lower()).strip("-")
    return f"{product}-{slug}"


def find_pdfs(settings: Settings, product: str | None = None) -> list[tuple[str, Path]]:
    products = [product] if product else list(settings.products)
    found: list[tuple[str, Path]] = []
    for key in products:
        folder = settings.raw_dir / key
        if folder.is_dir():
            found.extend((key, pdf) for pdf in sorted(folder.glob("*.pdf")))
    return found


def build_chunks(
    settings: Settings,
    product: str | None = None,
    log: Callable[[str], None] = print,
) -> list[Chunk]:
    sources = load_sources(settings.sources_path)
    chunks: list[Chunk] = []
    for key, pdf in find_pdfs(settings, product):
        info = sources.get((key, pdf.name))
        if info is None:
            log(f"Varoitus: {pdf.name} puuttuu tiedostosta sources.yaml, käytetään tiedostonimeä.")
        doc_chunks = chunk_document(
            read_pdf(pdf),
            product=key,
            document=info.title if info else pdf.stem,
            version=info.version if info else "",
            doc_id=_doc_id(key, pdf),
            max_chars=settings.chunking.max_chars,
            max_heading_chars=settings.chunking.max_heading_chars,
            label=info.label if info else "",
        )
        log(f"{key}/{pdf.name}: {len(doc_chunks)} palaa")
        chunks.extend(doc_chunks)
    return chunks


def write_index(settings: Settings, chunks: list[Chunk], embed: Callable[[list[str]], list[list[float]]]) -> None:
    """Replace the documents' chunks in the persistent Chroma collection.

    Chunks of each re-ingested document are deleted first so removed or renumbered
    sections do not linger in the index.
    """
    import chromadb

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.index_dir))
    collection = client.get_or_create_collection(settings.collection, metadata={"hnsw:space": "cosine"})

    for document in {(c.product, c.document) for c in chunks}:
        collection.delete(where={"$and": [{"product": document[0]}, {"document": document[1]}]})

    batch_size = settings.embed_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata() for c in batch],
            embeddings=embed([c.text for c in batch]),
        )
