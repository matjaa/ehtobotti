import csv
import dataclasses

from ehtobotti import evaluation
from ehtobotti.config import load_settings
from ehtobotti.retrieve import RetrievedChunk
from tests.test_generate import SETTINGS, FakeRetriever, reply


def chunk(section, part="", page=5):
    return RetrievedChunk(
        text=f"{section} Otsikko\nTeksti.",
        metadata={"product": "auto", "document": "d", "section": section, "section_title": "Otsikko",
                  "page": page, "part": part},
        score=0.8,
    )


def question(**overrides):
    base = dict(id="q1", product="auto", question="Kysymys?", expected_answer="Vastaus.",
                expected_sections=["3.9"], answerable=True)
    return evaluation.Question(**{**base, **overrides})


class ScriptedClient:
    """Answers questions with a fixed reply and judges with a fixed verdict."""

    def __init__(self, answer, verdict="OIKEIN"):
        self.answer, self.verdict = answer, verdict
        self.models = []

    def chat(self, messages, *, model=None, format=None, **opts):
        self.models.append(model)
        return self.verdict if messages[0]["role"] == "user" else self.answer

    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_section_matching_accepts_subsections_only():
    assert evaluation.section_matches("3.16.5", "3.16")
    assert evaluation.section_matches("3.16", "3.16")
    assert not evaluation.section_matches("3.1", "3.16")
    assert not evaluation.section_matches("3", "3.16")
    assert not evaluation.section_matches("", "3.16")


def test_expected_part_disambiguates_repeated_numbers():
    q = question(expected_sections=["4.1"], expected_part="Liikennevakuutusehdot")
    wrong_part = FakeRetriever([chunk("4.1", part="Vapaaehtoisen vakuutuksen ehdot")])
    right_part = FakeRetriever([chunk("4.1", part="Liikennevakuutusehdot")])
    client = ScriptedClient(reply(True, "Vastaus [Autovakuutus, kohta 4.1, s. 5]."))
    assert evaluation.run([q], retriever=wrong_part, client=client, settings=SETTINGS,
                          model="m", judge_model=None, log=lambda _: None)[0].hit is False
    assert evaluation.run([q], retriever=right_part, client=client, settings=SETTINGS,
                          model="m", judge_model=None, log=lambda _: None)[0].hit is True


def test_full_run_scores_and_summary():
    settings = dataclasses.replace(SETTINGS, products={"auto": "Autovakuutus"})
    questions = [
        question(id="ok"),
        question(id="none", expected_sections=[], answerable=False, expected_answer="-"),
    ]
    retriever = FakeRetriever([chunk("3.9"), chunk("3.10")])
    client = ScriptedClient(reply(True, "Omavastuu on 300 euroa [Autovakuutus, kohta 3.9, s. 5]."))
    results = evaluation.run(questions, retriever=retriever, client=client, settings=settings,
                             model="answer-model", judge_model="judge-model", log=lambda _: None)

    ok, none = results
    assert ok.hit is True
    assert ok.citation_precision == 1.0 and ok.citation_hit is True
    assert ok.judge_verdict == "OIKEIN" and ok.judge_score == 1.0
    assert none.refusal_correct is False  # answered an unanswerable question
    assert none.judge_score is None
    # All answers are produced before any judging, so models are not swapped per question.
    assert client.models == ["answer-model", "answer-model", "judge-model"]

    summary = evaluation.summarize(results)
    assert summary["hit_at_k"] == 1.0
    assert summary["refusal_accuracy"] == 0.0
    assert summary["answer_correctness"] == 1.0
    md = evaluation.summary_markdown(summary, model="answer-model", retrieval="vector",
                                     judge_model="judge-model", top_k=5)
    assert "| Haun osuvuus (hit@5) | 100 % |" in md


def test_refusal_on_answerable_question_scores_zero_without_judge_call():
    client = ScriptedClient(reply(False))
    [r] = evaluation.run([question()], retriever=FakeRetriever([chunk("3.9")]), client=client,
                         settings=SETTINGS, model="m", judge_model="j", log=lambda _: None)
    assert r.answer.refused and r.judge_verdict == "KIELTÄYTYI" and r.judge_score == 0.0
    assert client.models == ["m"]


def test_parse_verdict():
    assert evaluation.parse_verdict("OIKEIN") == "OIKEIN"
    assert evaluation.parse_verdict("Arvio: osittain oikein") == "OSITTAIN"
    assert evaluation.parse_verdict("VÄÄRIN, koska ...") == "VÄÄRIN"
    assert evaluation.parse_verdict("en tiedä") == ""


def test_results_files_have_windows_safe_names(tmp_path):
    [r] = evaluation.run([question()], retriever=FakeRetriever([chunk("3.9")]),
                         client=ScriptedClient(reply(True, "x")), settings=SETTINGS, model="qwen3.5:4b",
                         judge_model=None, log=lambda _: None)
    path = evaluation.write_results([r], "summary", tmp_path, "qwen3.5:4b", "vector")
    assert ":" not in path.name and "qwen3.5-4b_vector" in path.name
    assert path.with_suffix(".md").read_text(encoding="utf-8") == "summary"
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["id"] == "q1" and rows[0]["hit"] == "True"


def test_judge_is_never_the_answering_model():
    settings = load_settings()
    assert settings.judge_for(settings.judge_model) != settings.judge_model
    assert settings.judge_for(settings.default_chat_model) != settings.default_chat_model


def test_question_set_is_valid():
    questions = evaluation.load_questions(load_settings().questions_path)
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.answerable == q.has_expected, q.id
        for source in q.expected_sources:
            assert evaluation.parse_source(source)[0], q.id
        assert q.product in load_settings().products, q.id


def ipid_chunk(page):
    return RetrievedChunk(
        text="Mitä vakuutus ei kata?\nLiikennevakuutus ei kata omaa ajoneuvoa",
        metadata={"product": "auto", "document": "IPID", "section": "", "section_title": "Mitä vakuutus ei kata?",
                  "page": page, "label": "Autovakuutus IPID"},
        score=0.6,
    )


def test_parse_source():
    assert evaluation.parse_source("Autovakuutus IPID, s. 1") == ("Autovakuutus IPID", 1)
    assert evaluation.parse_source("Autovakuutus IPID") == ("Autovakuutus IPID", None)


def test_expected_sources_count_for_retrieval_and_citations():
    q = question(expected_sections=["3.9"], expected_sources=["Autovakuutus IPID, s. 1"],
                 expected_part="Vapaaehtoisen vakuutuksen ehdot")
    client = ScriptedClient(reply(False, ""))
    [r] = evaluation.run([q], retriever=FakeRetriever([ipid_chunk(1)]), client=client, settings=SETTINGS,
                         model="m", judge_model=None, log=lambda _: None)
    assert r.hit is True  # the IPID page alone is enough

    [r] = evaluation.run([q], retriever=FakeRetriever([ipid_chunk(2)]), client=client, settings=SETTINGS,
                         model="m", judge_model=None, log=lambda _: None)
    assert r.hit is False  # wrong page

    cited = ScriptedClient(reply(True, "Ei [Autovakuutus IPID, s. 1]. Lasiturva [Autovakuutus, kohta 3.9, s. 5]. "
                                       "Muu [Autovakuutus IPID, s. 2]."))
    [r] = evaluation.run([q], retriever=FakeRetriever([ipid_chunk(1)]), client=cited, settings=SETTINGS,
                         model="m", judge_model=None, log=lambda _: None)
    assert r.citation_hit is True
    assert r.citation_precision == 2 / 3


def test_run_limits_retrieval_to_the_questions_company():
    retriever = FakeRetriever([chunk("3.9")])
    client = ScriptedClient(reply(True, "Korvataan [Autovakuutus, kohta 3.9, s. 5]."), "OIKEIN")
    evaluation.run([question(company="LähiTapiola")], retriever=retriever, client=client, settings=SETTINGS,
                   model="test-model", judge_model=None)
    assert retriever.calls[-1]["company"] == "LähiTapiola"
