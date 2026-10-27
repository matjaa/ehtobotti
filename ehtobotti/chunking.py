"""Split insurance terms text into section-based chunks.

Insurance terms are numbered documents ("4 Vahinkotapahtumat", "4.2 Vuotovahingot").
Splitting on that numbering keeps each chunk about one topic and gives every chunk
a section number that the answer can cite. Sections longer than the size limit are
split further on paragraph boundaries, and the section heading is repeated at the
start of every sub-chunk so the retriever and the LLM always see the context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "4", "4.2", "4.2.1" optionally followed by a dot, then the title on the same line.
# Each level is 1-2 digits so years ("2026") and amounts ("100 euroa") never match.
_HEADING_RE = re.compile(r"^(?P<num>[1-9]\d?(?:\.\d{1,2}){0,3})\.?\s+(?P<title>\S.*)$")
# A bare section number on its own line; some PDFs put the title on the next line.
_BARE_NUMBER_RE = re.compile(r"^(?P<num>[1-9]\d?(?:\.\d{1,2}){0,3})\.?$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_TOC_LEADER_RE = re.compile(r"(\.{3,}|…|\s{2,}\d+$|\t\d+$)")

# Optional per-line layout hints from the PDF reader (see pdftext.py).
STYLE_HEADING = "heading"  # bold or larger font: may be a section heading
STYLE_PART = "part"  # large unnumbered heading that starts a new part of the document


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based page number as printed in citations
    text: str
    # One entry per line of `text`. None means no layout info: any line may be a heading.
    styles: tuple[str | None, ...] | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    product: str
    document: str
    version: str
    section: str
    section_title: str
    page: int
    chunk_id: str
    part: str = ""
    label: str = ""  # citation name for the document (sources.yaml), empty = product name

    def metadata(self) -> dict[str, str | int]:
        return {
            "product": self.product,
            "document": self.document,
            "version": self.version,
            "section": self.section,
            "section_title": self.section_title,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "part": self.part,
            "label": self.label,
        }


@dataclass
class _Line:
    text: str
    page: int
    style: str | None = None
    has_styles: bool = False

    @property
    def may_be_heading(self) -> bool:
        return not self.has_styles or self.style in (STYLE_HEADING, STYLE_PART)


@dataclass
class _Section:
    number: str
    title: str
    page: int
    part: str = ""
    lines: list[_Line] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}".strip()

    @property
    def prefix(self) -> str:
        """Lines repeated at the top of every chunk of this section.

        The part name is included so both the retriever and the LLM can tell e.g.
        "Liikennevakuutusehdot / 3 Vakuutuksen sisältö" from the voluntary insurance terms.
        """
        lines = [self.part] if self.part and self.part != self.title else []
        if self.heading:
            lines.append(self.heading)
        return "\n".join(lines)


def _is_title(title: str, number: str, max_heading_chars: int) -> bool:
    """Heuristics that separate real headings from numbered list items in body text."""
    title = title.strip()
    if not title or len(number) + 1 + len(title) > max_heading_chars:
        return False
    if not title[0].isupper():
        return False
    # Table of contents rows: "4.2 Vuotovahingot ........ 12"
    if _TOC_LEADER_RE.search(title):
        return False
    # Headings are labels, list items are sentences.
    return not title.endswith((".", ",", ";", ":"))


def _parse_heading(line: _Line, next_line: _Line | None, max_heading_chars: int) -> tuple[str, str, bool] | None:
    """Return (number, title, consumed_next_line) if `line` starts a section."""
    if not line.may_be_heading:
        return None
    match = _HEADING_RE.match(line.text)
    if match and _is_title(match["title"], match["num"], max_heading_chars):
        return match["num"], match["title"].strip(), False

    bare = _BARE_NUMBER_RE.match(line.text)
    if (
        bare
        and next_line
        and next_line.may_be_heading
        and _is_title(next_line.text, bare["num"], max_heading_chars)
    ):
        return bare["num"], next_line.text.strip(), True
    return None


def split_sections(pages: list[PageText], max_heading_chars: int = 90) -> list[_Section]:
    """Group the document's lines into sections by their numbered headings.

    Text before the first heading (cover page, table of contents intro) becomes a
    section with an empty number so nothing is silently dropped. A part heading
    (layout hint STYLE_PART) starts an unnumbered section titled with the part name,
    and every following section records which part it belongs to.
    """
    lines = [
        _Line(
            text=raw.strip(),
            page=p.page,
            style=p.styles[i] if p.styles is not None and i < len(p.styles) else None,
            has_styles=p.styles is not None,
        )
        for p in pages
        for i, raw in enumerate(p.text.splitlines())
    ]

    sections: list[_Section] = []
    current = _Section(number="", title="", page=lines[0].page if lines else 1)
    part = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        parsed = _parse_heading(line, next_line, max_heading_chars) if line.text else None
        if parsed:
            number, title, consumed = parsed
            i += 2 if consumed else 1
            # Long titles wrap onto the next line(s); with layout hints we can tell
            # a styled continuation ("... korjaaminen ja" + "lunastaminen") from body text.
            while (
                i < len(lines)
                and lines[i].has_styles
                and lines[i].style == STYLE_HEADING
                and lines[i].text
                and not _parse_heading(lines[i], lines[i + 1] if i + 1 < len(lines) else None, max_heading_chars)
                and len(number) + len(title) + len(lines[i].text) + 2 <= max_heading_chars
            ):
                title = f"{title} {lines[i].text}"
                i += 1
            sections.append(current)
            current = _Section(number=number, title=title, page=line.page, part=part)
            continue
        if line.style == STYLE_PART and line.text:
            part = line.text
            sections.append(current)
            current = _Section(number="", title=part, page=line.page, part=part)
            i += 1
            continue
        if line.style == STYLE_HEADING and line.text.endswith("?") and len(line.text) <= max_heading_chars:
            # Unnumbered question headings, as in the EU standard product information
            # document (IPID): "Mitä vakuutus kattaa?", "Missä vakuutus on voimassa?"
            sections.append(current)
            current = _Section(number="", title=line.text, page=line.page, part=part)
            i += 1
            continue
        current.lines.append(line)
        i += 1
    sections.append(current)

    return [s for s in sections if s.number or any(l.text for l in s.lines)]


def _paragraphs(lines: list[_Line]) -> list[list[_Line]]:
    """Split on blank lines; PDFs without blank lines fall back to one line per unit."""
    paragraphs: list[list[_Line]] = []
    current: list[_Line] = []
    for line in lines:
        if line.text:
            current.append(line)
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)

    if len(paragraphs) == 1 and len(paragraphs[0]) > 1:
        return [[line] for line in paragraphs[0]]
    return paragraphs


def _split_long_text(text: str, budget: int) -> list[str]:
    """Split a single oversized paragraph on sentence ends, then on whitespace."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END_RE.split(text):
        while len(sentence) > budget:
            cut = sentence.rfind(" ", 0, budget)
            cut = cut if cut > 0 else budget
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > budget and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _section_parts(section: _Section, max_chars: int) -> list[tuple[str, int]]:
    """Return (body text, start page) pieces whose text plus heading fits max_chars."""
    budget = max(max_chars - len(section.prefix) - 1, 1)
    parts: list[tuple[str, int]] = []
    current_text = ""
    current_page = section.page

    for paragraph in _paragraphs(section.lines):
        para_text = "\n".join(l.text for l in paragraph)
        para_page = paragraph[0].page

        if len(para_text) > budget:
            if current_text:
                parts.append((current_text, current_page))
                current_text = ""
            parts.extend((piece, para_page) for piece in _split_long_text(para_text, budget))
            continue

        candidate = f"{current_text}\n{para_text}" if current_text else para_text
        if len(candidate) > budget and current_text:
            parts.append((current_text, current_page))
            current_text, current_page = para_text, para_page
        else:
            if not current_text:
                current_page = para_page
            current_text = candidate

    if current_text or not parts:
        parts.append((current_text, current_page if current_text else section.page))
    return parts


def chunk_document(
    pages: list[PageText],
    *,
    product: str,
    document: str,
    version: str,
    doc_id: str,
    max_chars: int,
    max_heading_chars: int = 90,
    label: str = "",
) -> list[Chunk]:
    """Turn a document's pages into chunks with citation metadata."""
    chunks: list[Chunk] = []
    for section in split_sections(pages, max_heading_chars):
        for body, page in _section_parts(section, max_chars):
            text = f"{section.prefix}\n{body}".strip() if section.prefix else body.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    product=product,
                    document=document,
                    version=version,
                    section=section.number,
                    section_title=section.title,
                    page=page,
                    chunk_id=f"{doc_id}:{len(chunks):04d}",
                    part=section.part,
                    label=label,
                )
            )
    return chunks


def format_chunk_debug(chunk: Chunk) -> str:
    """Human-readable dump used by `ingest --show-chunks` to inspect splitting by eye."""
    header = (
        f"=== {chunk.chunk_id} | {chunk.product} | {chunk.part or '-'} | kohta {chunk.section or '-'} "
        f"{chunk.section_title} | s. {chunk.page} | {len(chunk.text)} merkkiä ==="
    )
    return f"{header}\n{chunk.text}\n"
