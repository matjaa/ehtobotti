import json

import pytest

from ehtobotti.generate import REFUSAL_TEXT
from ehtobotti.retrieve import IndexNotFoundError, RetrievedChunk
from ehtobotti.ui import PENDING_TEXT, Assistant, build_app, product_choices
from tests.test_generate import SETTINGS

CHUNK = RetrievedChunk(
    text="Vapaaehtoisen vakuutuksen ehdot\n3.9 Lasiturva\nOmavastuu on 300 € <ei kumppani>.",
    metadata={"product": "auto", "section": "3.9", "section_title": "Lasiturva", "page": 5,
              "part": "Vapaaehtoisen vakuutuksen ehdot", "label": "Autovakuutus"},
    score=0.81,
)


class FakeRetriever:
    def __init__(self, chunks=None, error=None):
        self.chunks = [CHUNK] if chunks is None else chunks
        self.error = error
        self.calls = []
        self.companies_searched = []

    def search(self, query, *, top_k=None, product=None, company=None):
        self.calls.append(product)
        self.companies_searched.append(company)
        if self.error:
            raise self.error
        return self.chunks


class FakeClient:
    def __init__(self, answerable=True, text="Omavastuu on 300 euroa [Autovakuutus, kohta 3.9, s. 5]."):
        self.reply = json.dumps({"vastattavissa": answerable, "vastaus": text if answerable else ""})
        self.models = []

    def chat(self, messages, *, model=None, format=None, **opts):
        self.models.append(model)
        return self.reply

    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_product_choices_have_no_all_option():
    assert product_choices(SETTINGS) == [("Kotivakuutus", "koti"), ("Autovakuutus", "auto")]


def test_respond_appends_answer_with_expandable_sources():
    retriever, client = FakeRetriever(), FakeClient()
    history, cleared = Assistant(SETTINGS, client, retriever).respond("Paljonko omavastuu on?", [], "auto", "")

    assert cleared == ""
    assert history[0] == {"role": "user", "content": "Paljonko omavastuu on?"}
    reply = history[1]["content"]
    assert "Omavastuu on 300 euroa" in reply
    assert "<details><summary>[Autovakuutus, kohta 3.9, s. 5] Lasiturva</summary>" in reply
    assert "&lt;ei kumppani&gt;" in reply  # excerpt text is escaped, not rendered as HTML
    assert "Lähteet" in reply
    assert retriever.calls == ["auto"]
    assert client.models == ["test-model"]  # empty model choice falls back to the default


OTHER = RetrievedChunk(
    text="Liikennevakuutusehdot\n3 Mitä korvataan\nOmaa ajoneuvoa ei korvata.",
    metadata={"product": "auto", "section": "3", "section_title": "Mitä korvataan", "page": 23,
              "part": "Liikennevakuutusehdot", "label": "Autovakuutus"},
    score=0.6,
)


def test_uncited_excerpts_go_to_collapsed_section():
    history, _ = Assistant(SETTINGS, FakeClient(), FakeRetriever(chunks=[OTHER, CHUNK])).respond("Omavastuu?", [])
    reply = history[1]["content"]
    sources, others = reply.split("Muut haetut otteet (1)")
    assert "kohta 3.9, s. 5" in sources and "kohta 3, s. 23" not in sources
    assert "kohta 3, s. 23" in others


def test_answer_without_citations_has_no_sources_heading():
    client = FakeClient(text="Omavastuu on 300 euroa.")
    history, _ = Assistant(SETTINGS, client, FakeRetriever()).respond("Omavastuu?", [])
    reply = history[1]["content"]
    assert "Lähteet" not in reply
    assert "Muut haetut otteet (1)" in reply


def test_refusal_shows_retrieved_excerpts_as_not_containing_answer():
    history, _ = Assistant(SETTINGS, FakeClient(answerable=False), FakeRetriever()).respond("Hinta?", None)
    reply = history[1]["content"]
    assert REFUSAL_TEXT in reply
    assert "eivät sisältäneet vastausta" in reply


def test_unsupported_citation_is_flagged():
    client = FakeClient(text="Korvataan [Autovakuutus, kohta 9.9, s. 40].")
    history, _ = Assistant(SETTINGS, client, FakeRetriever()).respond("Korvataanko?", [])
    reply = history[1]["content"]
    assert "Autovakuutus kohta 9.9 s. 40" in reply
    # FakeRetriever has no section lookup, so the cited text cannot be shown.
    assert "Viitattu, mutta ei haettu" in reply and "todennäköisesti keksitty" in reply


def _chunk(section, page, part, text="teksti"):
    return RetrievedChunk(text=text, metadata={"product": "auto", "section": section, "section_title": "Otsikko",
                                               "page": page, "part": part, "label": "Autovakuutus"}, score=0.0)


class IndexedRetriever(FakeRetriever):
    """Retrieves CHUNK only, but the whole "index" can be looked up by section."""

    index = [
        _chunk("3.4", 3, "Vapaaehtoisen vakuutuksen ehdot", "Varkausturva korvaa anastuksen."),
        _chunk("3.4", 24, "Liikennevakuutusehdot"),
        _chunk("4.6.4", 20, "Vapaaehtoisen vakuutuksen ehdot", "Omavastuu 20 %."),
    ]

    def get_section(self, section, page):
        return [c for c in self.index if c.section == section and (section or c.page == page)]


def test_cited_section_outside_top_k_is_shown_from_index():
    client = FakeClient(text="Korvataan [Autovakuutus, kohta 3.4, s. 3].")
    reply = Assistant(SETTINGS, client, IndexedRetriever()).respond("Varkaus?", [])[0][1]["content"]
    box = reply.split("Viitattu, mutta ei haettu")[1]
    assert "Varkausturva korvaa anastuksen." in box
    assert "s. 24" not in box  # same number in the other part, far from the cited page
    assert "osuvuus" not in box


def test_cited_subsection_falls_back_to_parent_section():
    client = FakeClient(text="Omavastuu [Autovakuutus, kohta 4.6.4.1, s. 20].")
    reply = Assistant(SETTINGS, client, IndexedRetriever()).respond("Omavastuu?", [])[0][1]["content"]
    assert "Omavastuu 20 %." in reply.split("Viitattu, mutta ei haettu")[1]


def test_empty_question_is_ignored():
    history, _ = Assistant(SETTINGS, FakeClient(), FakeRetriever()).respond("   ", [])
    assert history == []


def test_missing_index_is_explained_in_chat():
    assistant = Assistant(SETTINGS, FakeClient(), FakeRetriever(error=IndexNotFoundError("x")))
    history, _ = assistant.respond("Kysymys?", [])
    assert "python -m ehtobotti ingest" in history[1]["content"]


def test_format_sources_empty_when_nothing_retrieved():
    history, _ = Assistant(SETTINGS, FakeClient(), FakeRetriever(chunks=[])).respond("Kysymys?", [])
    assert "Lähteet" not in history[1]["content"]


def test_build_app_creates_blocks():
    pytest.importorskip("gradio")
    assert build_app(SETTINGS, Assistant(SETTINGS, FakeClient(), FakeRetriever())) is not None


def test_submit_shows_question_and_placeholder_before_answer():
    assistant = Assistant(SETTINGS, FakeClient(), FakeRetriever())
    history, cleared, pending = assistant.submit(" Omavastuu? ", [])
    assert cleared == "" and pending == "Omavastuu?"
    assert history[0]["content"] == "Omavastuu?"
    assert history[1]["content"] == PENDING_TEXT

    history = assistant.answer(pending, history)
    assert len(history) == 2
    assert "Omavastuu on 300 euroa" in history[1]["content"]


class ProductRetriever(FakeRetriever):
    def has_product(self, product, company=None):
        return product == "auto" and company in (None, "Yhtiö A")

    def companies(self, product=None):
        return ["Yhtiö A"] if product in (None, "auto") else []


def test_warning_for_product_without_documents():
    client = FakeClient()
    assistant = Assistant(SETTINGS, client, ProductRetriever())
    assert assistant.product_warning("auto") == ""
    assert assistant.product_warning("") == ""
    warning = assistant.product_warning("koti")
    assert "Kotivakuutus" in warning and "Lisää ehtoja" in warning

    # Asking anyway gives the warning without a model call.
    history, _ = assistant.respond("Korvataanko vuoto?", [], "koti")
    assert history[1]["content"] == warning
    assert client.models == []


def test_warning_without_upload_points_to_ingest():
    assistant = Assistant(SETTINGS, FakeClient(), ProductRetriever())
    assistant.allow_upload = False
    assert "python -m ehtobotti ingest" in assistant.product_warning("koti")


def test_warning_when_index_is_missing():
    assistant = Assistant(SETTINGS, FakeClient(), None)
    assistant._ready = lambda: (_ for _ in ()).throw(IndexNotFoundError("x"))
    assert "Kotivakuutus" in assistant.product_warning("koti")


def test_company_filter_and_choices():
    retriever = ProductRetriever()
    assistant = Assistant(SETTINGS, FakeClient(), retriever)
    assert assistant.companies("auto") == ["Yhtiö A"] and assistant.companies("koti") == []
    assert assistant.default_product() == "auto"
    assert "Yhtiö B: Autovakuutus" in assistant.product_warning("auto", "Yhtiö B")

    assistant.respond("Omavastuu?", [], "auto", "Yhtiö A")
    assert retriever.companies_searched == ["Yhtiö A"]
