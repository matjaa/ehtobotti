"""Build the prompt, call the LLM and check the citations in its answer."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ehtobotti.config import Settings
from ehtobotti.llm import LLMClient
from ehtobotti.retrieve import RetrievedChunk, Retriever

REFUSAL_TEXT = "En löytänyt vastausta vakuutusehdoista."
DISCLAIMER = "Tämä ei ole virallinen korvauspäätös."

# Finnish prompt: small models answer noticeably better Finnish when instructed in Finnish.
SYSTEM_PROMPT = f"""Olet avustaja, joka vastaa kysymyksiin vakuutusehdoista.

Säännöt:
1. Vastaa vain alla annettujen otteiden perusteella. Älä käytä muuta tietoa äläkä arvaa.
2. Merkitse jokaisen väitteen perään lähde täsmälleen siinä muodossa kuin se on otteen otsikossa, esimerkiksi [Kotivakuutus, kohta 4.2, s. 12].
3. Jos otteet eivät sisällä vastausta, vastaa vain: "{REFUSAL_TEXT}"
4. Lopeta vastaus lyhyeen huomautukseen: "{DISCLAIMER}"

Vastaa suomeksi, selkeästi ja lyhyesti."""

# [Kotivakuutus, kohta 4.2, s. 12]  or, for text before the first section,  [Kotivakuutus, s. 1]
_CITATION_RE = re.compile(r"\[(?P<product>[^\[\],]+?),\s*(?:kohta\s+(?P<section>[\d.]+?)\.?,\s*)?s\.\s*(?P<page>\d+)\]")


@dataclass(frozen=True)
class Citation:
    product: str
    section: str
    page: int


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    model: str
    latency_s: float
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    # Citations that do not match any retrieved chunk: a sign the model made them up.
    unsupported_citations: list[Citation] = field(default_factory=list)


def citation_label(chunk: RetrievedChunk, settings: Settings) -> str:
    product = settings.products.get(chunk.product, chunk.product)
    if chunk.section:
        return f"[{product}, kohta {chunk.section}, s. {chunk.page}]"
    return f"[{product}, s. {chunk.page}]"


def build_messages(question: str, chunks: list[RetrievedChunk], settings: Settings) -> list[dict[str, str]]:
    excerpts = "\n\n".join(
        f"Ote {i} {citation_label(chunk, settings)}\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
    )
    user = f"Otteet vakuutusehdoista:\n\n{excerpts}\n\nKysymys: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(text: str) -> list[Citation]:
    return [
        Citation(product=m["product"].strip(), section=m["section"] or "", page=int(m["page"]))
        for m in _CITATION_RE.finditer(text)
    ]


def find_unsupported(citations: list[Citation], chunks: list[RetrievedChunk], settings: Settings) -> list[Citation]:
    known = {
        (settings.products.get(c.product, c.product), c.section, c.page)
        for c in chunks
    }
    return [c for c in citations if (c.product, c.section, c.page) not in known]


def is_refusal(text: str) -> bool:
    return REFUSAL_TEXT.rstrip(".").lower() in text.lower()


def answer_question(
    question: str,
    *,
    retriever: Retriever,
    client: LLMClient,
    settings: Settings,
    product: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
) -> Answer:
    model = model or settings.default_chat_model
    start = time.perf_counter()
    chunks = retriever.search(question, top_k=top_k, product=product)

    if not chunks:
        # Nothing to ground an answer on: refuse without calling the model.
        return Answer(
            text=f"{REFUSAL_TEXT} {DISCLAIMER}",
            sources=[],
            model=model,
            latency_s=time.perf_counter() - start,
            refused=True,
        )

    text = client.chat(build_messages(question, chunks, settings), model=model).strip()
    citations = extract_citations(text)
    return Answer(
        text=text,
        sources=chunks,
        model=model,
        latency_s=time.perf_counter() - start,
        refused=is_refusal(text),
        citations=citations,
        unsupported_citations=find_unsupported(citations, chunks, settings),
    )
