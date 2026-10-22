from pathlib import Path

from ehtobotti.config import ChatModelOptions, ChunkingSettings, Settings
from ehtobotti.generate import (
    DISCLAIMER,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    answer_question,
    build_messages,
    citation_label,
    extract_citations,
    is_refusal,
)
from ehtobotti.retrieve import RetrievedChunk

SETTINGS = Settings(
    provider="ollama",
    ollama_host="http://localhost:11434",
    default_chat_model="test-model",
    embedding_model="test-embed",
    chat_models={"test-model": ChatModelOptions(think_supported=True)},
    num_ctx=4096,
    temperature=0.1,
    chunking=ChunkingSettings(),
    top_k=5,
    embed_batch_size=32,
    raw_dir=Path("data/raw"),
    index_dir=Path("data/index"),
    sources_path=Path("data/sources.yaml"),
    collection="test",
    products={"koti": "Kotivakuutus", "auto": "Autovakuutus"},
)

LEAK = RetrievedChunk(
    text="4.2 Vuotovahingot\nVakuutuksesta korvataan äkillinen vuoto putkistosta.",
    metadata={"product": "koti", "document": "Kotivakuutusehdot", "section": "4.2",
              "section_title": "Vuotovahingot", "page": 12, "chunk_id": "koti-x:0003"},
    score=0.82,
)
INTRO = RetrievedChunk(
    text="Kotivakuutusehdot, voimassa 1.1.2026 alkaen",
    metadata={"product": "koti", "document": "Kotivakuutusehdot", "section": "",
              "section_title": "", "page": 1, "chunk_id": "koti-x:0000"},
    score=0.40,
)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, *, top_k=None, product=None):
        self.calls.append({"query": query, "top_k": top_k, "product": product})
        return self.chunks


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.messages = None
        self.model = None

    def chat(self, messages, *, model=None, **opts):
        self.messages, self.model = messages, model
        return self.reply

    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_citation_label_formats():
    assert citation_label(LEAK, SETTINGS) == "[Kotivakuutus, kohta 4.2, s. 12]"
    assert citation_label(INTRO, SETTINGS) == "[Kotivakuutus, s. 1]"


def test_prompt_contains_rules_excerpts_and_question():
    messages = build_messages("Korvataanko vuoto?", [LEAK], SETTINGS)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert REFUSAL_TEXT in SYSTEM_PROMPT and DISCLAIMER in SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Ote 1 [Kotivakuutus, kohta 4.2, s. 12]" in user
    assert "äkillinen vuoto putkistosta" in user
    assert user.endswith("Kysymys: Korvataanko vuoto?")


def test_extract_citations():
    text = (
        "Äkillinen vuoto korvataan [Kotivakuutus, kohta 4.2, s. 12]. "
        "Ehdot ovat voimassa 2026 alkaen [Kotivakuutus, s. 1]."
    )
    cites = extract_citations(text)
    assert [(c.product, c.section, c.page) for c in cites] == [
        ("Kotivakuutus", "4.2", 12),
        ("Kotivakuutus", "", 1),
    ]


def test_answer_with_valid_citation():
    client = FakeClient(f"Äkillinen vuoto korvataan [Kotivakuutus, kohta 4.2, s. 12]. {DISCLAIMER}")
    retriever = FakeRetriever([LEAK, INTRO])
    answer = answer_question("Korvataanko vuoto?", retriever=retriever, client=client,
                             settings=SETTINGS, product="koti")
    assert not answer.refused
    assert answer.model == "test-model"
    assert answer.sources == [LEAK, INTRO]
    assert len(answer.citations) == 1
    assert answer.unsupported_citations == []
    assert retriever.calls[0]["product"] == "koti"


def test_made_up_citation_is_flagged():
    client = FakeClient("Vuoto korvataan [Kotivakuutus, kohta 9.9, s. 40].")
    answer = answer_question("Korvataanko vuoto?", retriever=FakeRetriever([LEAK]),
                             client=client, settings=SETTINGS)
    assert [(c.section, c.page) for c in answer.unsupported_citations] == [("9.9", 40)]


def test_refusal_is_detected():
    client = FakeClient(f"{REFUSAL_TEXT} {DISCLAIMER}")
    answer = answer_question("Paljonko kuu painaa?", retriever=FakeRetriever([LEAK]),
                             client=client, settings=SETTINGS)
    assert answer.refused
    assert is_refusal("en löytänyt vastausta vakuutusehdoista")


def test_no_chunks_refuses_without_calling_model():
    client = FakeClient("should not be used")
    answer = answer_question("Mitä tahansa?", retriever=FakeRetriever([]), client=client, settings=SETTINGS)
    assert answer.refused
    assert client.messages is None
    assert REFUSAL_TEXT in answer.text


def test_model_override_is_passed_to_client():
    client = FakeClient("ok")
    answer_question("Kysymys?", retriever=FakeRetriever([LEAK]), client=client,
                    settings=SETTINGS, model="other-model")
    assert client.model == "other-model"
