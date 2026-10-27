"""Build the prompt, call the LLM and check the citations in its answer.

The model replies in a fixed JSON shape: first a yes/no decision whether the excerpts
answer the question, then the answer. The code, not the model, decides what the user
sees when the answer is no (a fixed refusal sentence, no citations) and always appends
the disclaimer. This makes refusals reliable to detect and keeps their wording fixed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ehtobotti.config import Settings
from ehtobotti.llm import LLMClient
from ehtobotti.retrieve import RetrievedChunk, Retriever

REFUSAL_TEXT = "En löytänyt vastausta vakuutusehdoista."
DISCLAIMER = "Tämä ei ole virallinen korvauspäätös."

# Finnish prompt: small models answer noticeably better Finnish when instructed in Finnish.
SYSTEM_PROMPT = """Olet avustaja, joka vastaa kysymyksiin vakuutusehdoista pelkästään annettujen otteiden perusteella.

Vastaa JSON-muodossa:
- "vastattavissa": true, jos otteista löytyy vastaus kysymykseen. Myös kielteinen vastaus ("ei korvata") tai ehdollinen vastaus ("korvataan, jos ...") on vastaus. false, jos otteet eivät kerro asiasta.
- "vastaus": lyhyt vastaus lähdeviitteineen. Tyhjä, jos "vastattavissa" on false.

Ohjeet:
1. Älä arvaa äläkä käytä muuta tietoa. Jos otteista ei löydy vastausta, aseta "vastattavissa": false. Älä kirjoita vastaukseen, että otteet eivät kerro jotain.
2. Vastaa siihen, mitä kysytään, samassa muodossa kuin esimerkeissä.
3. Otteen ensimmäisellä rivillä lukee, mitä vakuutusta se koskee, esimerkiksi "Liikennevakuutusehdot" tai "Vapaaehtoisen vakuutuksen ehdot". Käytä vain otteita, jotka koskevat kysyttyä vakuutusta.
4. Merkitse lähde jokaisen väitteen perään hakasulkeissa täsmälleen kuten otteen otsikossa.

Esimerkkejä:

Kysymys: Korvataanko äkillinen putkivuoto?
{"vastattavissa": true, "vastaus": "Kyllä, äkillinen vuoto putkistosta korvataan [Kotivakuutus, kohta 4.2, s. 12]."}

Kysymys: Korvataanko vähitellen tapahtunut vuoto?
{"vastattavissa": true, "vastaus": "Ei. Vähitellen tapahtunutta vuotoa ei korvata [Kotivakuutus, kohta 4.3, s. 13]."}

Kysymys: Paljonko murtovahingon omavastuu on?
{"vastattavissa": true, "vastaus": "Omavastuu on 150 euroa [Kotivakuutus, kohta 9.1, s. 30]."}

Kysymys: Korvataanko polkupyörän varkaus?
{"vastattavissa": true, "vastaus": "Kyllä, jos pyörä on ollut lukittuna runkolukolla [Kotivakuutus, kohta 5.4, s. 16]."}

Kysymys: Korvaako kotivakuutuksen vastuuturva oman puhelimen rikkoutumisen?
{"vastattavissa": true, "vastaus": "Ei. Vastuuturva korvaa vain toiselle aiheutettuja vahinkoja [Kotivakuutus, kohta 7.1, s. 22]. Oman puhelimen rikkoutuminen korvataan irtaimistoturvasta [Kotivakuutus, kohta 4.1, s. 11]."}

Kysymys: Paljonko vakuutus maksaa kuukaudessa?
{"vastattavissa": false, "vastaus": ""}"""

# Field order matters: the model commits to the decision before writing the answer.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "vastattavissa": {"type": "boolean"},
        "vastaus": {"type": "string"},
    },
    "required": ["vastattavissa", "vastaus"],
}

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
    # False if the model's reply was not the expected JSON and a fallback was used.
    structured: bool = True


def source_name(chunk: RetrievedChunk, settings: Settings) -> str:
    """Name shown in citations: the document's label from sources.yaml, else the product name.

    A product can have several documents (terms, IPID), so the page number alone is
    ambiguous without a per-document label.
    """
    return str(chunk.metadata.get("label") or settings.products.get(chunk.product, chunk.product))


def citation_label(chunk: RetrievedChunk, settings: Settings) -> str:
    name = source_name(chunk, settings)
    if chunk.section:
        return f"[{name}, kohta {chunk.section}, s. {chunk.page}]"
    return f"[{name}, s. {chunk.page}]"


def build_messages(question: str, chunks: list[RetrievedChunk], settings: Settings) -> list[dict[str, str]]:
    # The excerpt header is exactly the citation label, so the model can copy it as is.
    excerpts = "\n\n---\n\n".join(f"{citation_label(chunk, settings)}\n{chunk.text}" for chunk in chunks)
    user = f"Otteet vakuutusehdoista:\n\n{excerpts}\n\nKysymys: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(text: str) -> list[Citation]:
    return [
        Citation(product=m["product"].strip(), section=m["section"] or "", page=int(m["page"]))
        for m in _CITATION_RE.finditer(text)
    ]


def find_unsupported(citations: list[Citation], chunks: list[RetrievedChunk], settings: Settings) -> list[Citation]:
    known = {(source_name(c, settings), c.section, c.page) for c in chunks}

    def supported(c: Citation) -> bool:
        if (c.product, c.section, c.page) in known:
            return True
        # A sub-subsection heading inside a chunk ("4.6.4.1" in chunk 4.6.4) is quoted from
        # the excerpt, not made up; the chunker only splits on headings it recognises.
        return any(
            name == c.product and section and c.section.startswith(section + ".") and abs(page - c.page) <= 1
            for name, section, page in known
        )

    return [c for c in citations if not supported(c)]


def is_refusal(text: str) -> bool:
    return REFUSAL_TEXT.rstrip(".").lower() in text.lower()


def parse_reply(reply: str) -> tuple[bool, str, bool]:
    """Return (answerable, answer text, structured) from the model's JSON reply.

    Falls back to treating the reply as plain text if it is not valid JSON
    (e.g. a provider that ignores the schema).
    """
    try:
        data = json.loads(reply)
        answerable, text = data["vastattavissa"], data["vastaus"]
        if not isinstance(answerable, bool) or not isinstance(text, str):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        text = reply.strip()
        # An empty reply (e.g. a thinking model that spent its context on reasoning) is no answer.
        return bool(text) and not is_refusal(text), text, False
    text = text.replace(DISCLAIMER, "").strip()
    # A "yes" with no answer text is not an answer.
    return answerable and bool(text), text, True


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

    reply = client.chat(build_messages(question, chunks, settings), model=model, format=ANSWER_SCHEMA)
    answerable, text, structured = parse_reply(reply)
    if not answerable:
        return Answer(
            text=f"{REFUSAL_TEXT} {DISCLAIMER}",
            sources=chunks,
            model=model,
            latency_s=time.perf_counter() - start,
            refused=True,
            structured=structured,
        )

    citations = extract_citations(text)
    return Answer(
        text=f"{text}\n\n{DISCLAIMER}",
        sources=chunks,
        model=model,
        latency_s=time.perf_counter() - start,
        refused=False,
        citations=citations,
        unsupported_citations=find_unsupported(citations, chunks, settings),
        structured=structured,
    )
