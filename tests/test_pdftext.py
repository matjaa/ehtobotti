from ehtobotti.chunking import STYLE_HEADING, STYLE_PART, PageText, chunk_document, split_sections
from ehtobotti.pdftext import RawLine, clean_pages, normalize_line


def body(text, y=0.5, start=False):
    return RawLine(text=text, y=y, size=9.0, bold=False, block_start=start)


def bold(text, y=0.5, start=True):
    return RawLine(text=text, y=y, size=9.0, bold=True, block_start=start)


def footer(page):
    return [RawLine(f"{page}", 0.94, 15.0, True, True), RawLine("LähiTapiola", 0.94, 8.0, True, True)]


def test_normalize_hair_spaced_section_numbers():
    assert normalize_line("3 . 20 . 2\tSijaisauto") == "3.20.2 Sijaisauto"
    assert normalize_line("3 . 10\tPetosturva") == "3.10 Petosturva"
    assert normalize_line("Voimassa 1.1.2026 alkaen") == "Voimassa 1.1.2026 alkaen"


def test_running_footer_is_removed():
    raw = [[bold("3.1"), bold("Kolarointi", start=False), body("Teksti sivulla.", start=True), *footer(n)]
           for n in range(1, 6)]
    pages = clean_pages(raw)
    assert all("LähiTapiola" not in p.text for p in pages)
    assert all(not p.text.splitlines()[-1].isdigit() for p in pages)


def test_toc_page_is_blanked():
    toc = [body(f"{i}.1 Otsikko {i}........ {i}", start=True) for i in range(1, 7)]
    pages = clean_pages([toc, [bold("1"), bold("Voimassaoloalue", start=False), body("Teksti.", start=True)]])
    assert pages[0].text == ""
    assert pages[1].text.startswith("1\nVoimassaoloalue")


def test_soft_hyphens_are_joined_and_bullets_attached():
    raw = [[bold("2.1"), body("Ajo­", start=True), body("neuvo on kohde."), body("•", start=True), body("lisävarusteet")]]
    text = clean_pages(raw)[0].text
    assert "Ajoneuvo on kohde." in text
    assert "• lisävarusteet" in text


def test_numbered_list_items_in_regular_font_are_not_sections():
    raw = [[bold("2.1"), bold("Ajoneuvo ja sen varusteet", start=False),
            body("1. Vakuutuksen kohteena on vakuutuskirjassa yksilöity", start=True),
            body("2 Vakuutuksen kohteena on myös", start=True)]]
    sections = split_sections(clean_pages(raw))
    assert [(s.number, s.title) for s in sections] == [("2.1", "Ajoneuvo ja sen varusteet")]


def test_parts_restart_numbering_and_are_recorded():
    part = RawLine("Liikennevakuutusehdot", 0.3, 13.0, False, True)
    raw = [
        [bold("4.1"), bold("Korvattavat vahingot", start=False), body("Teksti A.", start=True)],
        [part, bold("4.1"), bold("Vakuutuksenottajan tiedonantovelvollisuus", start=False),
         body("Teksti B.", start=True)],
    ]
    chunks = chunk_document(clean_pages(raw), product="auto", document="d", version="", doc_id="x", max_chars=800)
    by_text = {c.text.splitlines()[-1]: c for c in chunks}
    assert by_text["Teksti A."].part == ""
    assert by_text["Teksti B."].part == "Liikennevakuutusehdot"
    assert by_text["Teksti B."].section == "4.1"
    assert by_text["Teksti B."].metadata()["part"] == "Liikennevakuutusehdot"


def test_wrapped_heading_title_is_joined():
    pages = [PageText(page=1, text="4.6 Vakuutuksen kohteen korjaaminen ja\nlunastaminen\nTeksti.",
                      styles=(STYLE_HEADING, STYLE_HEADING, None))]
    assert split_sections(pages)[0].title == "Vakuutuksen kohteen korjaaminen ja lunastaminen"


def test_part_heading_alone_starts_unnumbered_section():
    pages = [PageText(page=2, text="Johdanto\nTämä on johdanto.", styles=(STYLE_PART, None))]
    sections = split_sections(pages)
    assert [(s.number, s.title, s.part) for s in sections] == [("", "Johdanto", "Johdanto")]


def test_wingdings_check_marks_become_symbols():
    from ehtobotti.pdftext import _span_text
    assert _span_text({"font": "Wingdings-Regular", "text": "ü Lasiturva"}) == "✓ Lasiturva"
    assert _span_text({"font": "Wingdings-Regular", "text": "û irtain omaisuus"}) == "✗ irtain omaisuus"
    assert _span_text({"font": "Sans-Regular", "text": "ü"}) == "ü"


def test_question_headings_start_unnumbered_sections():
    pages = [PageText(page=1, text="Mitä vakuutus kattaa?\n✓ Lasiturva\nMitä vakuutus ei kata?\n✗ Irtaimisto",
                      styles=(STYLE_HEADING, None, STYLE_HEADING, None))]
    assert [(s.number, s.title) for s in split_sections(pages)] == [
        ("", "Mitä vakuutus kattaa?"), ("", "Mitä vakuutus ei kata?")]
    # Without layout hints (plain text) question lines stay body text.
    plain = [PageText(page=1, text="Mitä vakuutus kattaa?\nLasiturva")]
    assert [s.title for s in split_sections(plain)] == [""]


def test_part_name_is_repeated_in_every_chunk():
    part = RawLine("Liikennevakuutusehdot", 0.3, 13.0, False, True)
    raw = [[part, bold("3"), bold("Vakuutuksen sisältö", start=False), body("Korvataan henkilövahinko.", start=True)]]
    [chunk] = chunk_document(clean_pages(raw), product="auto", document="d", version="", doc_id="x", max_chars=800)
    assert chunk.text.startswith("Liikennevakuutusehdot\n3 Vakuutuksen sisältö\n")
