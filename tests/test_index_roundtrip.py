"""Write chunks to a real Chroma index and search them back, with a fake embedder.

Checks the ingest/retrieve wiring (metadata, product filter, re-ingest) without Ollama.
"""

import dataclasses
import hashlib

import pytest

pytest.importorskip("chromadb")

from ehtobotti.chunking import PageText, chunk_document  # noqa: E402
from ehtobotti.ingest import write_index  # noqa: E402
from ehtobotti.retrieve import IndexNotFoundError, VectorRetriever  # noqa: E402
from tests.test_generate import SETTINGS  # noqa: E402

DIM = 64


def fake_embed(texts):
    """Bag-of-words hashing: texts sharing words get similar vectors."""
    vectors = []
    for text in texts:
        vec = [0.0] * DIM
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        vectors.append(vec)
    return vectors


def _chunks(product, pages, doc_id):
    return chunk_document(pages, product=product, document=f"{product} ehdot", version="2026",
                          doc_id=doc_id, max_chars=1600)


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(SETTINGS, index_dir=tmp_path / "index", embed_batch_size=2)


def test_missing_index_raises(settings):
    with pytest.raises(IndexNotFoundError):
        VectorRetriever(settings, fake_embed)


def test_roundtrip_with_product_filter(settings):
    koti = _chunks("koti", [PageText(1, "4.2 Vuotovahingot\nputkiston vuoto korvataan\n\n"
                                        "4.3 Murtovahingot\nmurto asuntoon korvataan\n")], "koti-a")
    auto = _chunks("auto", [PageText(3, "2.1 Lasivahingot\ntuulilasin vuoto korvataan\n")], "auto-a")
    write_index(settings, koti + auto, fake_embed)

    retriever = VectorRetriever(settings, fake_embed)
    top = retriever.search("putkiston vuoto", top_k=1)
    assert top[0].section == "4.2"
    assert top[0].page == 1
    assert top[0].document == "koti ehdot"

    only_auto = retriever.search("vuoto", top_k=5, product="auto")
    assert {c.product for c in only_auto} == {"auto"}
    assert only_auto[0].section == "2.1"


def test_reingest_replaces_old_chunks(settings):
    old = _chunks("koti", [PageText(1, "4.2 Vuotovahingot\nvanha teksti\n\n5 Poistuva kohta\npoistuu\n")], "koti-a")
    write_index(settings, old, fake_embed)
    new = _chunks("koti", [PageText(1, "4.2 Vuotovahingot\nuusi teksti\n")], "koti-a")
    write_index(settings, new, fake_embed)

    results = VectorRetriever(settings, fake_embed).search("teksti", top_k=10)
    assert [c.section for c in results] == ["4.2"]
    assert "uusi" in results[0].text
