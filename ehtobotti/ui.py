"""Gradio chat UI. User-facing text is in Finnish.

The UI is a thin layer over `answer_question`: the same retrieval, prompt and
citation checks as the CLI, so eval results describe what the UI does too.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ehtobotti.config import Settings
from ehtobotti.generate import Answer, Citation, answer_question, citation_label, find_unsupported, source_name
from ehtobotti.llm import LLMClient, get_client
from ehtobotti.retrieve import IndexNotFoundError, RetrievedChunk, Retriever, VectorRetriever

INTRO = """# Ehtobotti
Kysy vakuutusehdoista. Vastaus perustuu julkisiin vakuutusehtoihin, ja jokaisen väitteen
perässä on lähde (kohta ja sivu). Lähteiden tekstin näet vastauksen alta.

**Epävirallinen demo**, ei minkään vakuutusyhtiön palvelu. Vastaukset eivät ole virallisia
korvauspäätöksiä, eikä botti tunne omaa vakuutuskirjaasi.
Botti ei osaa kertoa suppean ja laajan vakuutuksen eroja.
Älä kirjoita kysymykseen henkilötietoja."""

PENDING_TEXT = "_Haetaan vastausta ehdoista…_"

EXAMPLES = [
    "Korvataanko eläintörmäysturvasta vahinko, joka syntyy kun väistän eläintä?",
    "Paljonko lasiturvan omavastuu on?",
    "Saako auton avaimia säilyttää autossa?",
]


def product_choices(settings: Settings) -> list[tuple[str, str]]:
    """Dropdown choices as (label, value). There is no "all" choice on purpose: terms of
    different insurances or companies mixed in one answer would mislead."""
    return [(name, key) for key, name in settings.products.items()]


def _source_item(chunk: RetrievedChunk, settings: Settings, show_score: bool = True) -> str:
    part = chunk.metadata.get("part") or chunk.document
    summary = html.escape(f"{citation_label(chunk, settings)} {chunk.section_title}".strip())
    detail = html.escape(f"{part} · osuvuus {chunk.score:.2f}" if show_score else part)
    excerpt = html.escape(chunk.text).replace("\n", "<br>")
    return f"<details><summary>{summary}</summary><p><em>{detail}</em></p><blockquote>{excerpt}</blockquote></details>"


def _citation_text(c: Citation) -> str:
    return " ".join(filter(None, [c.product, c.section and f"kohta {c.section}", f"s. {c.page}"]))


def lookup_citation(citation: Citation, retriever: Retriever, settings: Settings) -> list[RetrievedChunk]:
    """Find the indexed text of a cited section that was not among the retrieved chunks.

    Section numbers restart in each part of the terms, so the chunks closest to the
    cited page win. A sub-subsection that is not a chunk of its own ("4.6.4.1") falls
    back to its parent section on the same or next page.
    """
    get_section = getattr(retriever, "get_section", None)
    if get_section is None:
        return []
    section, exact = citation.section, True
    while True:
        found = [c for c in get_section(section, citation.page) if source_name(c, settings) == citation.product]
        if not exact:
            found = [c for c in found if abs(c.page - citation.page) <= 1]
        if found:
            nearest = min(abs(c.page - citation.page) for c in found)
            return [c for c in found if abs(c.page - citation.page) == nearest]
        if "." not in section:
            return []
        section, exact = section.rsplit(".", 1)[0], False


def format_unretrieved(lookups: list[tuple[Citation, list[RetrievedChunk]]], settings: Settings) -> str:
    """Box for citations outside the retrieved excerpts: the text the model did not see."""
    items = []
    for citation, chunks in lookups:
        if chunks:
            items.extend(_source_item(c, settings, show_score=False) for c in chunks)
        else:
            items.append(f"<p>{html.escape(_citation_text(citation))}: kohtaa ei löydy ehdoista, "
                         "joten viite on todennäköisesti keksitty.</p>")
    note = "Malli ei nähnyt näitä otteita vastatessaan. Tarkista, pitääkö väite paikkansa."
    return f"**Viitattu, mutta ei haettu:**\n\n<em>{note}</em>\n\n" + "\n".join(items)


def is_cited(chunk: RetrievedChunk, answer: Answer, settings: Settings) -> bool:
    # Same matching as the unsupported-citation check, so the two never disagree.
    return any(not find_unsupported([c], [chunk], settings) for c in answer.citations)


def format_sources(answer: Answer, settings: Settings) -> str:
    """Expandable source lists (HTML <details>) with section, page and excerpt text.

    "Lähteet" holds only the excerpts the answer cites. The rest of the retrieved
    excerpts sit in one closed section: noise for most users, but they show what the
    model had to work with (e.g. the right section was retrieved but not used).
    """
    if not answer.sources:
        return ""
    if answer.refused:
        items = "\n".join(_source_item(c, settings) for c in answer.sources)
        return f"**Haetut otteet (eivät sisältäneet vastausta):**\n\n{items}"
    cited = [c for c in answer.sources if is_cited(c, answer, settings)]
    others = [c for c in answer.sources if c not in cited]
    blocks = []
    if cited:
        blocks.append("**Lähteet:**\n\n" + "\n".join(_source_item(c, settings) for c in cited))
    if others:
        inner = "\n".join(_source_item(c, settings) for c in others)
        blocks.append(f"<details><summary><strong>Muut haetut otteet ({len(others)})</strong></summary><blockquote>{inner}</blockquote></details>")
    return "\n\n".join(blocks)


def format_answer(
    answer: Answer, settings: Settings, lookups: list[tuple[Citation, list[RetrievedChunk]]] | None = None
) -> str:
    parts = [answer.text]
    if answer.unsupported_citations:
        bad = ", ".join(_citation_text(c) for c in answer.unsupported_citations)
        parts.append(f"⚠️ Vastauksessa on viitteitä, joita ei löydy haetuista otteista: {bad}")
    sources = format_sources(answer, settings)
    if sources:
        parts.append(sources)
    if lookups:
        parts.append(format_unretrieved(lookups, settings))
    parts.append(f"<sub>Malli: {html.escape(answer.model)} · aika {answer.latency_s:.1f} s</sub>")
    return "\n\n".join(parts)


class Assistant:
    """Holds the model client and retriever; created on the first question so the UI
    starts even when Ollama or the index is not ready yet."""

    def __init__(self, settings: Settings, client: LLMClient | None = None, retriever: Retriever | None = None) -> None:
        self.settings = settings
        self._client = client
        self._retriever = retriever
        # Decides what the empty-product warning tells the user to do.
        self.allow_upload = True

    def _ready(self) -> tuple[LLMClient, Retriever]:
        if self._client is None:
            self._client = get_client(self.settings)
        if self._retriever is None:
            self._retriever = VectorRetriever(self.settings, self._client.embed)
        return self._client, self._retriever

    def _has(self, product: str, company: str | None = None) -> bool:
        try:
            _, retriever = self._ready()
        except IndexNotFoundError:
            return False  # no index at all: nothing has documents
        has_product = getattr(retriever, "has_product", None)
        return True if has_product is None else has_product(product, company)

    def companies(self, product: str | None = None) -> list[str]:
        """Companies with indexed documents (for one product, or all)."""
        try:
            _, retriever = self._ready()
        except IndexNotFoundError:
            return []
        companies = getattr(retriever, "companies", None)
        return companies(product) if companies else []

    def default_product(self) -> str:
        """First insurance that has documents, so the chat opens ready to use."""
        return next((p for p in self.settings.products if self._has(p)), next(iter(self.settings.products), ""))

    def product_warning(self, product: str, company: str = "") -> str:
        """Warning shown when the chosen insurance (and company) has no indexed documents, else ""."""
        if not product or self._has(product, company or None):
            return ""
        name = self.settings.products.get(product, product)
        if company:
            name = f"{company}: {name}"
        how = ("Lisää PDF välilehdellä **Lisää ehtoja**." if self.allow_upload
               else f"Lisää PDF kansioon `data/raw/{product}/<yhtiö>/` ja aja `python -m ehtobotti ingest`.")
        return f"⚠️ Vakuutukselle **{name}** ei ole lisätty yhtään ehtoasiakirjaa, joten botti ei löydä siitä vastauksia. {how}"

    def reply(self, question: str, product: str = "", company: str = "", model: str = "") -> str:
        # Answering from an empty product would only ever give a refusal; say why instead.
        warning = self.product_warning(product, company)
        if warning:
            return warning
        try:
            client, retriever = self._ready()
            answer = answer_question(
                question, retriever=retriever, client=client, settings=self.settings,
                product=product or None, company=company or None, model=model or None,
            )
        except IndexNotFoundError:
            return "Hakuindeksiä ei löytynyt. Aja ensin: `python -m ehtobotti ingest`"
        except ConnectionError:
            return f"Ollamaan ei saatu yhteyttä osoitteessa {self.settings.ollama_host}. Onko `ollama serve` käynnissä?"
        # dict.fromkeys: the same made-up citation may appear several times in one answer.
        lookups = [(c, lookup_citation(c, retriever, self.settings)) for c in dict.fromkeys(answer.unsupported_citations)]
        return format_answer(answer, self.settings, lookups)

    def add_document(
        self, file_path: str | None, product: str, company: str = "", title: str = "", label: str = "",
        version: str = "",
    ) -> str:
        """Upload handler: save and index one terms PDF, return a status message."""
        from ehtobotti.ingest import add_document

        if not file_path:
            return "Valitse ensin PDF-tiedosto."
        if not product:
            return "Valitse, mitä vakuutusta asiakirja koskee."
        if not (company or "").strip():
            return "Anna vakuutusyhtiön nimi."
        try:
            if self._client is None:
                self._client = get_client(self.settings)
            info, count, existed = add_document(
                self.settings, product, Path(file_path), self._client.embed,
                company=company, title=title, version=version, label=label,
            )
        except ValueError as exc:
            return f"Tiedostoa ei lisätty: {exc}"
        except ConnectionError:
            return f"Ollamaan ei saatu yhteyttä osoitteessa {self.settings.ollama_host}. Onko `ollama serve` käynnissä?"
        # Reopen the index on the next question, so a first-ever upload is found too.
        self._retriever = None
        note = " Tiedosto oli jo lisätty, joten se korvattiin ja tiedot pidettiin ennallaan." if existed else ""
        return (f"Lisätty **{info.file}** ({info.company}, {self.settings.products[product]}): {count} palaa indeksoitu. "
                f"Lähdeviitteissä nimi on **{info.label}**.{note}")

    @staticmethod
    def submit(question: str, history: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], str, str]:
        """First half of a chat turn, instant: show the question and a placeholder answer,
        clear the input box. Returns (history, input box, question to answer)."""
        history = list(history or [])
        question = question.strip()
        if question:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": PENDING_TEXT})
        return history, "", question

    def answer(
        self, question: str, history: list[dict[str, Any]] | None, product: str = "", company: str = "",
        model: str = "",
    ) -> list[dict[str, Any]]:
        """Second half of a chat turn, slow: replace the placeholder with the answer."""
        history = list(history or [])
        if question:
            history[-1] = {"role": "assistant", "content": self.reply(question, product, company, model)}
        return history

    def respond(
        self, question: str, history: list[dict[str, Any]] | None, product: str = "", company: str = "",
        model: str = "",
    ) -> tuple[list[dict[str, Any]], str]:
        """Both halves in one call (CLI-style use and tests)."""
        history, cleared, question = self.submit(question, history)
        return self.answer(question, history, product, company, model), cleared


UPLOAD_INFO = """Lisää uusi vakuutusehtoasiakirja (PDF). Tiedosto tallennetaan kansioon
`data/raw/<tuote>/<yhtiö>/`, merkitään tiedostoon `data/sources.yaml` ja indeksoidaan heti.
Indeksointi kestää asiakirjan koosta riippuen noin minuutin.

**Lisää vain julkisia vakuutusehtoja.** Älä lisää omaa vakuutuskirjaa tai muita
henkilötietoja sisältäviä asiakirjoja."""


def build_app(settings: Settings, assistant: Assistant | None = None, allow_upload: bool = True):
    import gradio as gr  # imported lazily so the CLI and tests work without it

    assistant = assistant or Assistant(settings)
    assistant.allow_upload = allow_upload
    models = list(dict.fromkeys([settings.default_chat_model, *settings.chat_models]))
    start_product = assistant.default_product()
    start_companies = assistant.companies(start_product)

    def select_product(product: str):
        """Company choices follow the insurance: only companies with documents for it."""
        companies = assistant.companies(product)
        company = companies[0] if companies else None
        return gr.update(choices=companies, value=company), assistant.product_warning(product, company or "")

    # No usage analytics to Gradio's servers: the demo is meant to run fully locally.
    with gr.Blocks(title="Ehtobotti", analytics_enabled=False) as app:
        gr.Markdown(INTRO)
        with gr.Tab("Kysy"):
            with gr.Row():
                product = gr.Dropdown(product_choices(settings), value=start_product, label="Vakuutus")
                company = gr.Dropdown(start_companies, value=start_companies[0] if start_companies else None,
                                      label="Vakuutusyhtiö")
                model = gr.Dropdown(models, value=settings.default_chat_model, label="Kielimalli")
            product_warning = gr.Markdown(assistant.product_warning(
                start_product, start_companies[0] if start_companies else ""))
            chatbot = gr.Chatbot(label="Keskustelu", height=520)
            question = gr.Textbox(label="Kysymys", placeholder="Esim. Korvataanko tuulilasin rikkoutuminen?")
            with gr.Row():
                send = gr.Button("Kysy", variant="primary")
                clear = gr.Button("Tyhjennä")
            gr.Examples(EXAMPLES, inputs=question, label="Esimerkkikysymyksiä")

        # Two steps so the question shows at once and only the chat shows progress.
        # Gradio's default progress overlay covers every output component, which left
        # a grey box over the input field and an extra scrollbar while the model ran.
        pending = gr.State("")
        for trigger in (send.click, question.submit):
            trigger(assistant.submit, [question, chatbot], [chatbot, question, pending],
                    queue=False, show_progress="hidden").then(
                assistant.answer, [pending, chatbot, product, company, model], chatbot, show_progress="hidden")
        clear.click(lambda: ([], ""), None, [chatbot, question], queue=False, show_progress="hidden")
        product.change(select_product, product, [company, product_warning], show_progress="hidden")

        # Off when the app is shared publicly: uploads write files onto this machine.
        if allow_upload:
            with gr.Tab("Lisää ehtoja"):
                gr.Markdown(UPLOAD_INFO)
                pdf = gr.File(label="PDF-tiedosto", file_types=[".pdf"], type="filepath")
                with gr.Row():
                    upload_product = gr.Dropdown(product_choices(settings), value=None, label="Vakuutus")
                    # Free text, but existing names are offered so one insurer is not
                    # split in two by spelling ("LähiTapiola" vs "Lähitapiola").
                    upload_company = gr.Dropdown(assistant.companies(), value=None, allow_custom_value=True,
                                                 label="Vakuutusyhtiö")
                with gr.Row():
                    title = gr.Textbox(label="Asiakirjan nimi", placeholder="Esim. Kotivakuutusehdot")
                    label = gr.Textbox(label="Nimi lähdeviitteissä", placeholder="Oletus: yhtiö ja vakuutuksen nimi")
                    version = gr.Textbox(label="Versio tai voimaantulopäivä", placeholder="Esim. 1.1.2026")
                add = gr.Button("Lisää ja indeksoi", variant="primary")
                status = gr.Markdown()
                # Refresh the chat tab: the upload may have added a company or filled the chosen product.
                add.click(assistant.add_document, [pdf, upload_product, upload_company, title, label, version],
                          status).then(
                    select_product, product, [company, product_warning], show_progress="hidden").then(
                    lambda: gr.update(choices=assistant.companies()), None, upload_company, show_progress="hidden")
    return app
