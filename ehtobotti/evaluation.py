"""Evaluate retrieval and answers against a hand-checked question set.

Questions are answered first with one model, then judged with another, so only one
chat model needs to be in GPU memory at a time.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable

from ehtobotti.config import Settings
from ehtobotti.generate import Answer, answer_question, source_name
from ehtobotti.llm import LLMClient
from ehtobotti.retrieve import Retriever

JUDGE_PROMPT = """Arvioi, vastaako mallin vastaus samaa kuin oikea vastaus.
Älä arvioi kieltä, tyyliä tai lähdeviitteiden muotoa, vain sisällön oikeellisuutta.

Kysymys: {question}

Oikea vastaus: {expected}

Mallin vastaus: {answer}

Vastaa yhdellä sanalla:
OIKEIN, jos mallin vastaus sisältää oikean vastauksen olennaisen sisällön eikä ole ristiriidassa sen kanssa
OSITTAIN, jos vastaus on osittain oikein tai siitä puuttuu olennainen osa
VÄÄRIN, jos vastaus on väärä tai ristiriidassa oikean vastauksen kanssa"""

JUDGE_SCORES = {"OIKEIN": 1.0, "OSITTAIN": 0.5, "VÄÄRIN": 0.0}


@dataclass(frozen=True)
class Question:
    id: str
    product: str
    question: str
    expected_answer: str
    expected_sections: list[str]
    answerable: bool
    # Documents can contain several parts that restart numbering (e.g. "4.1" twice);
    # when set, a retrieved chunk only counts as a hit if it is from this part.
    expected_part: str = ""
    # True once the expected answer and sections have been checked against the PDF.
    verified: bool = False
    # Expected sources outside the numbered terms, as "<citation label>, s. <page>"
    # (e.g. "Autovakuutus IPID, s. 1") or just "<citation label>" for any page.
    # A retrieved chunk or citation matching any expected section OR source counts.
    expected_sources: list[str] = field(default_factory=list)
    # The insurer whose terms the expected answer comes from. Retrieval is limited to it,
    # so another company's documents of the same product cannot change the results.
    company: str = ""

    @property
    def has_expected(self) -> bool:
        return bool(self.expected_sections or self.expected_sources)


@dataclass
class Result:
    question: Question
    answer: Answer
    hit: bool | None = None
    citation_precision: float | None = None
    citation_hit: bool | None = None
    judge_verdict: str = ""
    judge_score: float | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def refusal_correct(self) -> bool:
        return self.answer.refused != self.question.answerable


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question(**item) for item in json.load(f)]


def section_matches(found: str, expected: str) -> bool:
    """A subsection counts as a match: expected 3.16 is satisfied by 3.16.5."""
    return bool(found) and (found == expected or found.startswith(expected + "."))


def parse_source(expected: str) -> tuple[str, int | None]:
    """"Autovakuutus IPID, s. 1" -> ("Autovakuutus IPID", 1); "Autovakuutus IPID" -> (..., None)."""
    match = re.fullmatch(r"\s*(.+?)(?:,\s*s\.\s*(\d+))?\s*", expected)
    name, page = match.group(1), match.group(2)
    return name, int(page) if page else None


def source_matches(name: str, page: int, expected: str) -> bool:
    expected_name, expected_page = parse_source(expected)
    return name == expected_name and (expected_page is None or page == expected_page)


def score_retrieval(q: Question, answer: Answer, settings: Settings) -> bool | None:
    if not q.answerable or not q.has_expected:
        return None
    return any(
        any(
            section_matches(chunk.section, expected)
            and (not q.expected_part or chunk.metadata.get("part", "") == q.expected_part)
            for expected in q.expected_sections
        )
        or any(source_matches(source_name(chunk, settings), chunk.page, s) for s in q.expected_sources)
        for chunk in answer.sources
    )


def score_citations(q: Question, answer: Answer) -> tuple[float | None, bool | None]:
    """(share of citations that are expected, whether any citation is expected)."""
    if not q.answerable or not q.has_expected or not answer.citations:
        return None, (False if q.answerable and q.has_expected and not answer.refused else None)
    correct = [
        any(section_matches(c.section, expected) for expected in q.expected_sections)
        or any(source_matches(c.product, c.page, s) for s in q.expected_sources)
        for c in answer.citations
    ]
    return sum(correct) / len(correct), any(correct)


def parse_verdict(text: str) -> str:
    upper = text.upper()
    found = [(upper.find(word), word) for word in JUDGE_SCORES if word in upper]
    return min(found)[1] if found else ""


def judge(result: Result, client: LLMClient, judge_model: str) -> None:
    q = result.question
    if not q.answerable:
        return
    if result.answer.refused:
        result.judge_verdict, result.judge_score = "KIELTÄYTYI", 0.0
        return
    prompt = JUDGE_PROMPT.format(question=q.question, expected=q.expected_answer, answer=result.answer.text)
    verdict = parse_verdict(client.chat([{"role": "user", "content": prompt}], model=judge_model))
    result.judge_verdict = verdict or "TUNTEMATON"
    result.judge_score = JUDGE_SCORES.get(verdict)


def run(
    questions: list[Question],
    *,
    retriever: Retriever,
    client: LLMClient,
    settings: Settings,
    model: str,
    judge_model: str | None,
    log: Callable[[str], None] = print,
) -> list[Result]:
    results: list[Result] = []
    for i, q in enumerate(questions, start=1):
        log(f"[{i}/{len(questions)}] {q.id}")
        answer = answer_question(q.question, retriever=retriever, client=client, settings=settings,
                                 product=q.product or None, company=q.company or None, model=model)
        result = Result(question=q, answer=answer, hit=score_retrieval(q, answer, settings))
        result.citation_precision, result.citation_hit = score_citations(q, answer)
        results.append(result)

    if judge_model:
        log(f"Arvioidaan vastaukset mallilla {judge_model}...")
        for result in results:
            judge(result, client, judge_model)
    return results


def _avg(values: list[float | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return mean(present) if present else None


def summarize(results: list[Result]) -> dict[str, float | int | None]:
    answerable = [r for r in results if r.question.answerable]
    unanswerable = [r for r in results if not r.question.answerable]
    return {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "verified": sum(r.question.verified for r in results),
        "hit_at_k": _avg([r.hit for r in answerable]),
        "citation_precision": _avg([r.citation_precision for r in answerable]),
        "citation_hit": _avg([r.citation_hit for r in answerable]),
        "answer_correctness": _avg([r.judge_score for r in answerable]),
        "refusal_accuracy": _avg([r.refusal_correct for r in unanswerable]),
        "false_refusal_rate": _avg([r.answer.refused for r in answerable]),
        "unsupported_citations": sum(len(r.answer.unsupported_citations) for r in results),
        "avg_latency_s": _avg([r.answer.latency_s for r in results]),
    }


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f} %"


def summary_markdown(summary: dict, *, model: str, retrieval: str, judge_model: str | None, top_k: int) -> str:
    latency = summary["avg_latency_s"]
    rows = [
        ("Kysymyksiä (vastattavissa / ei vastattavissa)",
         f"{summary['questions']} ({summary['answerable']} / {summary['unanswerable']})"),
        ("Tarkistettuja kysymyksiä", f"{summary['verified']} / {summary['questions']}"),
        (f"Haun osuvuus (hit@{top_k})", _pct(summary["hit_at_k"])),
        ("Lähdeviitteiden tarkkuus", _pct(summary["citation_precision"])),
        ("Vastauksia, joissa oikea lähdeviite", _pct(summary["citation_hit"])),
        ("Vastausten oikeellisuus (LLM-arvio)", _pct(summary["answer_correctness"])),
        ("Oikeat kieltäytymiset (ei vastattavissa)", _pct(summary["refusal_accuracy"])),
        ("Turhat kieltäytymiset (vastattavissa)", _pct(summary["false_refusal_rate"])),
        ("Keksityt lähdeviitteet (kpl)", str(summary["unsupported_citations"])),
        ("Keskimääräinen vastausaika", "–" if latency is None else f"{latency:.1f} s"),
    ]
    header = f"Malli: `{model}` · haku: `{retrieval}` · arvioija: `{judge_model or '–'}`\n\n"
    table = "| Mittari | Tulos |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    return header + table + "\n"


def _slug(text: str) -> str:
    # Windows forbids ":" in filenames, and model names contain it (qwen3.5:4b).
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


def write_results(results: list[Result], summary_md: str, results_dir: Path, model: str, retrieval: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(model)}_{retrieval}"
    csv_path = results_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows ä/ö
        writer = csv.writer(f)
        writer.writerow([
            "id", "product", "answerable", "verified", "expected_sections", "expected_sources", "retrieved_sections",
            "hit", "cited_sections", "citation_precision", "citation_hit", "unsupported_citations",
            "refused", "refusal_correct", "structured", "judge_verdict", "judge_score", "latency_s", "question", "answer",
        ])
        for r in results:
            q, a = r.question, r.answer
            writer.writerow([
                q.id, q.product, q.answerable, q.verified, ";".join(q.expected_sections), ";".join(q.expected_sources),
                ";".join(c.section or f"s. {c.page}" for c in a.sources),
                r.hit, ";".join(c.section or f"{c.product} s. {c.page}" for c in a.citations), r.citation_precision, r.citation_hit,
                len(a.unsupported_citations), a.refused, r.refusal_correct, a.structured, r.judge_verdict, r.judge_score,
                f"{a.latency_s:.2f}", q.question, a.text,
            ])
    (results_dir / f"{stem}.md").write_text(summary_md, encoding="utf-8")
    return csv_path
