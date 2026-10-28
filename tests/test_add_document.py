"""Add a single PDF the way the UI upload does: copy, record in sources.yaml, index."""

import dataclasses

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("chromadb")

from ehtobotti.ingest import add_document, load_sources  # noqa: E402
from ehtobotti.retrieve import VectorRetriever  # noqa: E402
from ehtobotti.ui import Assistant  # noqa: E402
from tests.test_generate import SETTINGS  # noqa: E402
from tests.test_index_roundtrip import fake_embed  # noqa: E402

SOURCES_HEADER = "# Public source documents.\n"


@pytest.fixture
def settings(tmp_path):
    sources = tmp_path / "sources.yaml"
    sources.write_text(SOURCES_HEADER, encoding="utf-8")
    return dataclasses.replace(SETTINGS, raw_dir=tmp_path / "raw", index_dir=tmp_path / "index",
                               sources_path=sources)


def make_pdf(path, lines):
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        # Headings are recognised by bold font, as in real terms PDFs.
        bold = line[:1].isdigit()
        page.insert_text((72, 72 + 20 * i), line, fontsize=11, fontname="hebo" if bold else "helv")
    doc.save(str(path))
    return path


def test_add_document_copies_records_and_indexes(settings, tmp_path):
    upload = make_pdf(tmp_path / "upload.pdf", ["4.2 Vuotovahingot", "putkiston vuoto korvataan"])

    info, count, existed = add_document(settings, "koti", upload, fake_embed, company="Lähi Yhtiö",
                                        filename="kotiehdot.pdf", title="Kotivakuutusehdot", version="2026")

    assert count >= 1 and not existed
    assert info.file == "lahi-yhtio/kotiehdot.pdf"
    assert (settings.raw_dir / "koti" / "lahi-yhtio" / "kotiehdot.pdf").exists()
    assert info.label == "Lähi Yhtiö Kotivakuutus"
    assert settings.sources_path.read_text(encoding="utf-8").startswith(SOURCES_HEADER)  # comments kept
    recorded = load_sources(settings.sources_path)[("koti", info.file)]
    assert recorded.title == "Kotivakuutusehdot" and recorded.company == "Lähi Yhtiö"
    retriever = VectorRetriever(settings, fake_embed)
    top = retriever.search("putkiston vuoto", top_k=1, product="koti", company="Lähi Yhtiö")
    assert top[0].section == "4.2"
    assert top[0].metadata["label"] == "Lähi Yhtiö Kotivakuutus"
    assert retriever.companies("koti") == ["Lähi Yhtiö"]
    assert retriever.has_product("koti", "Lähi Yhtiö") and not retriever.has_product("koti", "Muu")


def test_two_companies_are_kept_apart(settings, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = make_pdf(tmp_path / "a" / "ehdot.pdf", ["4.2 Vuotovahingot", "putkiston vuoto korvataan"])
    b = make_pdf(tmp_path / "b" / "ehdot.pdf", ["4.2 Vuotovahingot", "putkiston vuoto ei korvata"])
    add_document(settings, "koti", a, fake_embed, company="Yhtiö A")
    add_document(settings, "koti", b, fake_embed, company="Yhtiö B")  # same file name, other company

    retriever = VectorRetriever(settings, fake_embed)
    assert retriever.companies("koti") == ["Yhtiö A", "Yhtiö B"]
    only_b = retriever.search("vuoto", top_k=10, product="koti", company="Yhtiö B")
    assert {c.metadata["company"] for c in only_b} == {"Yhtiö B"}
    assert "ei korvata" in only_b[0].text


def test_second_document_of_product_gets_distinct_label(settings, tmp_path):
    first = make_pdf(tmp_path / "a.pdf", ["1.1 Yleista", "teksti"])
    second = make_pdf(tmp_path / "b.pdf", ["1.1 Muuta", "teksti"])
    add_document(settings, "koti", first, fake_embed, company="Yhtiö")
    info, _, _ = add_document(settings, "koti", second, fake_embed, company="Yhtiö")
    assert info.label == "Yhtiö Kotivakuutus (b)"


def test_reupload_keeps_metadata(settings, tmp_path):
    upload = make_pdf(tmp_path / "ehdot.pdf", ["1.1 Yleista", "teksti"])
    add_document(settings, "koti", upload, fake_embed, company="Yhtiö", title="Alkuperainen")
    info, _, existed = add_document(settings, "koti", upload, fake_embed, company="Yhtiö", title="Uusi")
    assert existed and info.title == "Alkuperainen"
    assert len(load_sources(settings.sources_path)) == 1


@pytest.mark.parametrize("name,content", [("ehdot.txt", b"%PDF-1.7"), ("ehdot.pdf", b"not a pdf")])
def test_rejects_non_pdf(settings, tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError):
        add_document(settings, "koti", path, fake_embed, company="Yhtiö")
    assert not settings.raw_dir.exists()


def test_rejects_pdf_without_text(settings, tmp_path):
    empty = make_pdf(tmp_path / "skannattu.pdf", [])
    with pytest.raises(ValueError, match="tekstiä"):
        add_document(settings, "koti", empty, fake_embed, company="Yhtiö")
    assert not settings.raw_dir.exists()


class EmbedClient:
    def embed(self, texts):
        return fake_embed(texts)


def test_ui_upload_reports_status(settings, tmp_path):
    upload = make_pdf(tmp_path / "kotiehdot.pdf", ["4.2 Vuotovahingot", "putkiston vuoto korvataan"])
    assistant = Assistant(settings, client=EmbedClient())
    assert "Lisätty **yhtio/kotiehdot.pdf** (Yhtiö, Kotivakuutus)" in assistant.add_document(
        str(upload), "koti", "Yhtiö")
    assert "Valitse" in assistant.add_document(None, "koti", "Yhtiö")
    assert "Valitse" in assistant.add_document(str(upload), "", "Yhtiö")
    assert "yhtiön nimi" in assistant.add_document(str(upload), "koti", " ")


def test_company_name_is_required(settings, tmp_path):
    upload = make_pdf(tmp_path / "ehdot.pdf", ["1.1 Yleista", "teksti"])
    with pytest.raises(ValueError, match="yhtiön"):
        add_document(settings, "koti", upload, fake_embed, company="  ")
