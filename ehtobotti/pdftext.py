"""Extract clean, layout-aware text from insurance terms PDFs.

Plain `page.get_text()` loses information the chunker needs, so this module reads
PyMuPDF's line/span structure instead and:
- marks lines set in bold or a larger font as heading candidates, so numbered list
  items in body text ("1. Vakuutuksen kohteena on ...") are not taken as sections
- marks large unnumbered headings as document parts ("Yleiset sopimusehdot"); one PDF
  can contain several parts that each restart their section numbering at 1
- removes running headers/footers (company name, page numbers)
- blanks out table of contents pages
- normalises section numbers typeset with hair spaces ("3 . 20" -> "3.20") and joins
  words hyphenated with soft hyphens at line ends
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from ehtobotti.chunking import STYLE_HEADING, STYLE_PART, PageText

_NUMBER_DOT_RE = re.compile(r"(?<=\d)[\s   ]*\.[\s   ]*(?=\d)")
_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_STARTS_WITH_NUMBER_RE = re.compile(r"^\d")
_BOLD_FONT_RE = re.compile(r"bold|semibold|extrab|black|heavy", re.IGNORECASE)
SOFT_HYPHEN = "\u00ad"
BULLETS = {"\u2022", "\u25aa", "\u25e6", "\u2013"}  # bullet, small square, white bullet, en dash

MARGIN = 0.08  # header/footer band, as a fraction of page height
REPEAT_SHARE = 0.3  # a margin line on at least this share of pages is a running header/footer
TOC_MIN_LINES = 5
PART_SIZE_RATIO = 1.3
HEADING_SIZE_RATIO = 1.1


@dataclass
class RawLine:
    text: str
    y: float  # top of the line as a fraction of page height
    size: float
    bold: bool
    block_start: bool  # first line of a PyMuPDF text block (paragraph-ish)


def normalize_line(text: str) -> str:
    text = _NUMBER_DOT_RE.sub(".", text)
    return " ".join(text.replace("\t", " ").split())


def _span_text(span: dict) -> str:
    # Symbol fonts map glyphs to Latin letters: Wingdings "ü" is a check mark, "û" a cross.
    if "wingdings" in span["font"].lower():
        return re.sub(r"^(\s*)ü", r"\1✓", re.sub(r"^(\s*)û", r"\1✗", span["text"]))
    return span["text"]


def _page_lines(page) -> list[RawLine]:
    height = page.rect.height or 1.0
    lines: list[RawLine] = []
    for block in page.get_text("dict")["blocks"]:
        for i, line in enumerate(block.get("lines", [])):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(_span_text(s) for s in line["spans"])
            main = max(spans, key=lambda s: len(s["text"]))
            lines.append(
                RawLine(
                    text=text,
                    y=line["bbox"][1] / height,
                    size=main["size"],
                    bold=bool(main["flags"] & 16) or bool(_BOLD_FONT_RE.search(main["font"])),
                    block_start=i == 0,
                )
            )
    return lines


def _margin_key(line: RawLine) -> tuple[str, float] | None:
    if MARGIN < line.y < 1 - MARGIN:
        return None
    return re.sub(r"\d+", "#", normalize_line(line.text)), round(line.y, 2)


def _remove_running_lines(pages: list[list[RawLine]]) -> list[list[RawLine]]:
    counts = Counter(key for page in pages for key in {_margin_key(l) for l in page} if key)
    threshold = max(3, REPEAT_SHARE * len(pages))
    repeated = {key for key, n in counts.items() if n >= threshold}
    return [[l for l in page if _margin_key(l) not in repeated] for page in pages]


def _join_soft_hyphens(lines: list[RawLine]) -> list[RawLine]:
    joined: list[RawLine] = []
    for line in lines:
        if joined and joined[-1].text.rstrip().endswith(SOFT_HYPHEN) and not line.block_start:
            prev = joined[-1]
            prev.text = prev.text.rstrip()[:-1] + line.text.lstrip()
            continue
        joined.append(RawLine(line.text, line.y, line.size, line.bold, line.block_start))
    for line in joined:
        line.text = line.text.replace(SOFT_HYPHEN, "")
    return joined


def _style(line: RawLine, text: str, body_size: float) -> str | None:
    if line.size >= body_size * PART_SIZE_RATIO and not _STARTS_WITH_NUMBER_RE.match(text):
        return STYLE_PART
    if line.bold or line.size >= body_size * HEADING_SIZE_RATIO:
        return STYLE_HEADING
    return None


def clean_pages(raw_pages: list[list[RawLine]]) -> list[PageText]:
    sizes = [l.size for page in raw_pages for l in page for _ in range(len(l.text))]
    body_size = median(sizes) if sizes else 10.0

    pages: list[PageText] = []
    for number, lines in enumerate(_remove_running_lines(raw_pages), start=1):
        if sum(bool(_TOC_LINE_RE.search(l.text)) for l in lines) >= TOC_MIN_LINES:
            pages.append(PageText(page=number, text=""))
            continue

        texts: list[str] = []
        styles: list[str | None] = []
        bullet = ""
        for line in _join_soft_hyphens(lines):
            text = normalize_line(line.text)
            if not text:
                continue
            if text in BULLETS:
                bullet = text  # bullet glyph is its own line; attach it to the item text
                continue
            if bullet:
                text, bullet = f"{bullet} {text}", ""
            if line.block_start and texts:
                texts.append("")  # blank line between blocks marks a paragraph break
                styles.append(None)
            texts.append(text)
            styles.append(_style(line, text, body_size))
        pages.append(PageText(page=number, text="\n".join(texts), styles=tuple(styles)))
    return pages


def read_pdf(path: Path) -> list[PageText]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return clean_pages([_page_lines(page) for page in doc])
