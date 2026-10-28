"""Build the vector index: PDF -> text -> section chunks -> embeddings -> Chroma."""

from __future__ import annotations

import re
import shutil
import unicodedata
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
    company: str = ""


def load_sources(path: Path) -> dict[tuple[str, str], SourceInfo]:
    """Map (product, file) to the document's public metadata.

    `file` is the path inside data/raw/<product>/, e.g. "lahitapiola/kotivakuutusehdot.pdf".
    """
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
            company=e.get("company", ""),
        )
        for e in entries
    }


def slugify(text: str) -> str:
    """Folder- and id-safe name: "LähiTapiola" -> "lahitapiola"."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def _doc_id(product: str, file: str) -> str:
    return f"{product}-{slugify(str(Path(file).with_suffix('')))}"


def find_pdfs(settings: Settings, product: str | None = None) -> list[tuple[str, Path]]:
    """PDFs under data/raw/<product>/, including company subfolders."""
    products = [product] if product else list(settings.products)
    found: list[tuple[str, Path]] = []
    for key in products:
        folder = settings.raw_dir / key
        if folder.is_dir():
            found.extend((key, pdf) for pdf in sorted(folder.rglob("*.pdf")))
    return found


def _source_file(settings: Settings, product: str, pdf: Path) -> str:
    return pdf.relative_to(settings.raw_dir / product).as_posix()


def _document_chunks(settings: Settings, key: str, pdf: Path, info: SourceInfo | None, file: str) -> list[Chunk]:
    """Chunk one PDF. `file` is its path inside data/raw/<product>/ (`pdf` may be a temporary copy)."""
    return chunk_document(
        read_pdf(pdf),
        product=key,
        document=info.title if info else Path(file).stem,
        version=info.version if info else "",
        doc_id=_doc_id(key, file),
        max_chars=settings.chunking.max_chars,
        max_heading_chars=settings.chunking.max_heading_chars,
        label=info.label if info else "",
        company=info.company if info else "",
    )


def build_chunks(
    settings: Settings,
    product: str | None = None,
    log: Callable[[str], None] = print,
) -> list[Chunk]:
    sources = load_sources(settings.sources_path)
    chunks: list[Chunk] = []
    for key, pdf in find_pdfs(settings, product):
        file = _source_file(settings, key, pdf)
        info = sources.get((key, file))
        if info is None:
            log(f"Varoitus: {file} puuttuu tiedostosta sources.yaml, käytetään tiedostonimeä.")
        doc_chunks = _document_chunks(settings, key, pdf, info, file)
        log(f"{key}/{file}: {len(doc_chunks)} palaa")
        chunks.extend(doc_chunks)
    return chunks


def _append_source(path: Path, info: SourceInfo) -> None:
    """Append an entry as text, so the comments in sources.yaml are kept."""
    entry = yaml.safe_dump([info.__dict__], allow_unicode=True, sort_keys=False)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + entry)


def add_document(
    settings: Settings,
    product: str,
    uploaded: Path,
    embed: Callable[[list[str]], list[list[float]]],
    *,
    company: str,
    filename: str | None = None,
    title: str = "",
    version: str = "",
    label: str = "",
    url: str = "",
) -> tuple[SourceInfo, int, bool]:
    """Add one PDF (e.g. uploaded from the UI): copy it to data/raw/<product>/<company>/,
    record it in sources.yaml and index it. Returns (source info, chunk count, was already known).

    A file that is already in sources.yaml keeps its recorded metadata; the PDF itself is
    replaced and re-indexed.
    """
    if product not in settings.products:
        raise ValueError(f"Tuntematon tuote: {product}")
    company = " ".join(company.split())
    if not slugify(company):
        raise ValueError("Anna vakuutusyhtiön nimi.")
    name = Path(filename or uploaded.name).name  # never trust a path from the browser
    if Path(name).suffix.lower() != ".pdf":
        raise ValueError("Vain PDF-tiedostot ovat tuettuja.")
    with uploaded.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("Tiedosto ei ole PDF.")

    # Each company gets its own folder, so same-named files of two insurers do not clash.
    file = f"{slugify(company)}/{name}"
    sources = load_sources(settings.sources_path)
    known = sources.get((product, file))
    if known:
        info = known
    else:
        used = {s.label or settings.products[s.product] for s in sources.values() if s.product == product}
        label = label.strip() or f"{company} {settings.products[product]}"
        if label in used:
            label = f"{label} ({Path(name).stem})"  # citations must tell the documents apart
        info = SourceInfo(product=product, file=file, title=title.strip() or Path(name).stem,
                          version=version.strip(), url=url.strip(), label=label, company=company)

    target = settings.raw_dir / product / file
    # Chunk before copying, so a PDF without extractable text never lands in data/raw.
    chunks = _document_chunks(settings, product, uploaded, info, file)
    if not chunks:
        raise ValueError("PDF:stä ei saatu tekstiä. Onko se skannattu kuva?")
    target.parent.mkdir(parents=True, exist_ok=True)
    if uploaded.resolve() != target.resolve():
        shutil.copyfile(uploaded, target)
    if not known:
        _append_source(settings.sources_path, info)
    write_index(settings, chunks, embed)
    return info, len(chunks), known is not None


def _doc_ids_by_product(chunks: list[Chunk]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for c in chunks:
        found.setdefault(c.product, set()).add(c.chunk_id.rsplit(":", 1)[0])
    return found


def write_index(settings: Settings, chunks: list[Chunk], embed: Callable[[list[str]], list[list[float]]]) -> None:
    """Replace the documents' chunks in the persistent Chroma collection.

    Chunks of each re-ingested document are deleted first so removed or renumbered
    sections do not linger in the index.
    """
    import chromadb

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.index_dir))
    collection = client.get_or_create_collection(settings.collection, metadata={"hnsw:space": "cosine"})

    # Chunk ids are "<doc_id>:<n>"; matching on the id also removes chunks indexed before
    # a metadata field (e.g. company) was added.
    for product, doc_ids in _doc_ids_by_product(chunks).items():
        existing = collection.get(where={"product": product}, include=[])["ids"]
        stale = [i for i in existing if i.rsplit(":", 1)[0] in doc_ids]
        if stale:
            collection.delete(ids=stale)

    batch_size = settings.embed_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata() for c in batch],
            embeddings=embed([c.text for c in batch]),
        )
