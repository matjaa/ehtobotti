import json
from pathlib import Path

from ehtobotti.config import ChatModelOptions, ChunkingSettings, Settings
from ehtobotti.generate import (
    ANSWER_SCHEMA,
    DISCLAIMER,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    answer_question,
    build_messages,
    citation_label,
    extract_citations,
    is_refusal,
    parse_reply,
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
    seed=None,
    chunking=ChunkingSettings(),
    top_k=5,
    embed_batch_size=32,
    raw_dir=Path("data/raw"),
    index_dir=Path("data/index"),
    sources_path=Path("data/sources.yaml"),
    collection="test",
    questions_path=Path("eval/questions.json"),
    results_dir=Path("results"),
    judge_model="judge-model",
    judge_fallback="fallback-judge",
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


def reply(answerable, text=""):
    """The JSON shape the model is constrained to."""
    return json.dumps({"vastattavissa": answerable, "vastaus": text}, ensure_ascii=False)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.messages = None
        self.model = None
        self.format = None

    def chat(self, messages, *, model=None, format=None, **opts):
        self.messages, self.model, self.format = messages, model, format
        return self.reply

    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_citation_label_formats():
    assert citation_label(LEAK, SETTINGS) == "[Kotivakuutus, kohta 4.2, s. 12]"
    assert citation_label(INTRO, SETTINGS) == "[Kotivakuutus, s. 1]"


def test_prompt_contains_rules_excerpts_and_question():
    messages = build_messages("Korvataanko vuoto?", [LEAK], SETTINGS)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert '"vastattavissa"' in SYSTEM_PROMPT
    user = messages[1]["content"]
    # Header is exactly the citation label so the model can copy it verbatim.
    assert "[Kotivakuutus, kohta 4.2, s. 12]\n4.2 Vuotovahingot" in user
    assert "Ote " not in user
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
    client = FakeClient(reply(True, "Äkillinen vuoto korvataan [Kotivakuutus, kohta 4.2, s. 12]."))
    retriever = FakeRetriever([LEAK, INTRO])
    answer = answer_question("Korvataanko vuoto?", retriever=retriever, client=client,
                             settings=SETTINGS, product="koti")
    assert not answer.refused and answer.structured
    assert client.format == ANSWER_SCHEMA
    assert answer.model == "test-model"
    assert answer.sources == [LEAK, INTRO]
    assert len(answer.citations) == 1
    assert answer.unsupported_citations == []
    assert retriever.calls[0]["product"] == "koti"


def test_disclaimer_is_added_by_code_exactly_once():
    answer = answer_question("Q?", retriever=FakeRetriever([LEAK]), settings=SETTINGS,
                             client=FakeClient(reply(True, f"Korvataan [Kotivakuutus, kohta 4.2, s. 12]. {DISCLAIMER}")))
    assert answer.text.endswith(DISCLAIMER)
    assert answer.text.count(DISCLAIMER) == 1


def test_made_up_citation_is_flagged():
    client = FakeClient(reply(True, "Vuoto korvataan [Kotivakuutus, kohta 9.9, s. 40]."))
    answer = answer_question("Korvataanko vuoto?", retriever=FakeRetriever([LEAK]),
                             client=client, settings=SETTINGS)
    assert [(c.section, c.page) for c in answer.unsupported_citations] == [("9.9", 40)]


def test_not_answerable_gives_fixed_refusal_without_citations():
    # Whatever the model writes in "vastaus", the user sees the fixed sentence.
    client = FakeClient(reply(False, "Otteissa ei mainita tätä [Kotivakuutus, kohta 4.2, s. 12]."))
    answer = answer_question("Paljonko kuu painaa?", retriever=FakeRetriever([LEAK]),
                             client=client, settings=SETTINGS)
    assert answer.refused
    assert answer.text == f"{REFUSAL_TEXT} {DISCLAIMER}"
    assert answer.citations == [] and answer.unsupported_citations == []


def test_answerable_with_empty_text_counts_as_refusal():
    answer = answer_question("Q?", retriever=FakeRetriever([LEAK]), settings=SETTINGS,
                             client=FakeClient(reply(True, "  ")))
    assert answer.refused


def test_parse_reply_falls_back_to_plain_text():
    assert parse_reply("Vuoto korvataan.") == (True, "Vuoto korvataan.", False)
    assert parse_reply(f"{REFUSAL_TEXT}")[0] is False
    assert parse_reply('{"vastattavissa": "yes", "vastaus": "x"}')[2] is False


def test_parse_reply_empty_reply_is_not_an_answer():
    assert parse_reply("") == (False, "", False)
    assert parse_reply("  \n")[0] is False
    assert is_refusal("en löytänyt vastausta vakuutusehdoista")


def test_no_chunks_refuses_without_calling_model():
    client = FakeClient("should not be used")
    answer = answer_question("Mitä tahansa?", retriever=FakeRetriever([]), client=client, settings=SETTINGS)
    assert answer.refused
    assert client.messages is None
    assert REFUSAL_TEXT in answer.text


def test_model_override_is_passed_to_client():
    client = FakeClient(reply(True, "ok"))
    answer_question("Kysymys?", retriever=FakeRetriever([LEAK]), client=client,
                    settings=SETTINGS, model="other-model")
    assert client.model == "other-model"


def test_document_label_is_used_in_citations():
    ipid = RetrievedChunk(
        text="Mitä vakuutus kattaa?\n✓ Lasiturva",
        metadata={"product": "koti", "document": "IPID", "section": "", "section_title": "Mitä vakuutus kattaa?",
                  "page": 1, "label": "Kotivakuutus IPID"},
        score=0.7,
    )
    assert citation_label(ipid, SETTINGS) == "[Kotivakuutus IPID, s. 1]"
    client = FakeClient(reply(True, "Lasiturva sisältyy [Kotivakuutus IPID, s. 1]. Ehdot [Kotivakuutus, s. 1]."))
    answer = answer_question("Q?", retriever=FakeRetriever([ipid]), client=client, settings=SETTINGS)
    # Same page number, different document: only the IPID citation is backed by a retrieved chunk.
    assert [(c.product, c.page) for c in answer.unsupported_citations] == [("Kotivakuutus", 1)]


def test_prompt_says_negative_answers_are_answerable():
    assert "Myös kielteinen vastaus" in SYSTEM_PROMPT and "ehdollinen vastaus" in SYSTEM_PROMPT


def test_quoted_sub_subsection_is_not_flagged_as_made_up():
    omavastuu = RetrievedChunk(
        text="4.6.4 Omavastuut\n4.6.4.1 Poikkeavat omavastuut\n20 %:n omavastuu",
        metadata={"product": "koti", "document": "d", "section": "4.6.4", "section_title": "Omavastuut", "page": 20},
        score=0.8,
    )
    client = FakeClient(reply(True, "20 % [Kotivakuutus, kohta 4.6.4.1, s. 20]. Väärä [Kotivakuutus, kohta 4.6.5, s. 20]."))
    answer = answer_question("Q?", retriever=FakeRetriever([omavastuu]), client=client, settings=SETTINGS)
    assert [c.section for c in answer.unsupported_citations] == ["4.6.5"]
