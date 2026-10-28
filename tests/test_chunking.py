from ehtobotti.chunking import PageText, chunk_document, split_sections


def _chunk(pages, max_chars=1600):
    return chunk_document(
        pages,
        product="koti",
        document="Kotivakuutusehdot",
        version="2026",
        doc_id="koti-test",
        max_chars=max_chars,
    )


SAMPLE = [
    PageText(
        page=1,
        text=(
            "Kotivakuutusehdot\n"
            "Voimassa 1.1.2026 alkaen\n"
            "\n"
            "4 Vahinkotapahtumat\n"
            "Tässä luvussa kerrotaan korvattavat vahingot.\n"
            "\n"
            "4.1 Palovahingot\n"
            "Vakuutuksesta korvataan tulipalon aiheuttama vahinko.\n"
        ),
    ),
    PageText(
        page=2,
        text=(
            "4.2 Vuotovahingot\n"
            "Vakuutuksesta korvataan äkillinen vuoto putkistosta.\n"
            "Ehdot:\n"
            "1. Vuoto on tapahtunut äkillisesti.\n"
            "2. Laitteisto on asennettu määräysten mukaisesti.\n"
        ),
    ),
]


def test_splits_on_numbered_headings():
    sections = split_sections(SAMPLE)
    assert [(s.number, s.title) for s in sections] == [
        ("", ""),
        ("4", "Vahinkotapahtumat"),
        ("4.1", "Palovahingot"),
        ("4.2", "Vuotovahingot"),
    ]


def test_metadata_and_pages():
    chunks = _chunk(SAMPLE)
    by_section = {c.section: c for c in chunks}
    assert by_section["4.1"].page == 1
    assert by_section["4.2"].page == 2
    assert by_section["4.2"].section_title == "Vuotovahingot"
    meta = by_section["4.2"].metadata()
    assert meta["product"] == "koti"
    assert meta["document"] == "Kotivakuutusehdot"
    assert meta["version"] == "2026"
    assert meta["chunk_id"].startswith("koti-test:")


def test_chunk_ids_are_unique():
    chunks = _chunk(SAMPLE)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_numbered_list_items_are_not_headings():
    chunks = _chunk(SAMPLE)
    assert {c.section for c in chunks} == {"", "4", "4.1", "4.2"}
    assert "1. Vuoto on tapahtunut äkillisesti." in next(c for c in chunks if c.section == "4.2").text


def test_dates_years_and_amounts_are_not_headings():
    pages = [PageText(page=1, text="3 Omavastuu\n150 euroa vahinkoa kohden\n2026 alkaen ehdot muuttuvat\n")]
    sections = split_sections(pages)
    assert [s.number for s in sections] == ["3"]


def test_table_of_contents_rows_are_not_headings():
    pages = [PageText(page=1, text="Sisällys\n4.2 Vuotovahingot ........ 12\n4.3 Murtovahingot …… 13\n")]
    assert [s.number for s in split_sections(pages)] == [""]


def test_heading_with_title_on_next_line():
    pages = [PageText(page=3, text="5.1\nRajoitukset\nVakuutus ei korvaa kulumista.\n")]
    sections = split_sections(pages)
    assert [(s.number, s.title) for s in sections] == [("5.1", "Rajoitukset")]


def test_long_section_is_split_with_heading_prefix_in_every_chunk():
    paragraphs = "\n\n".join(f"Kappale {i}. " + "Vakuutus korvaa vahingon. " * 8 for i in range(10))
    pages = [PageText(page=1, text=f"6.3 Pitkä kohta\n{paragraphs}\n")]
    chunks = _chunk(pages, max_chars=400)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("6.3 Pitkä kohta\n")
        assert chunk.section == "6.3"
        assert len(chunk.text) <= 400
    # No text lost: every paragraph marker survives the split.
    joined = " ".join(c.text for c in chunks)
    assert all(f"Kappale {i}." in joined for i in range(10))


def test_sub_chunk_page_follows_its_first_paragraph():
    first = "Ensimmäinen kappale. " * 15
    second = "Toinen kappale sivulla kaksi. " * 15
    pages = [
        PageText(page=1, text=f"7 Korvaus\n{first}\n"),
        PageText(page=2, text=f"\n{second}\n"),
    ]
    chunks = _chunk(pages, max_chars=400)
    assert chunks[0].page == 1
    assert any(c.page == 2 and "Toinen kappale" in c.text for c in chunks)


def test_single_oversized_paragraph_is_split_on_sentences():
    text = " ".join(f"Lause numero {i} kertoo ehdoista." for i in range(60))
    pages = [PageText(page=1, text=f"8 Yleiset ehdot\n{text}\n")]
    chunks = _chunk(pages, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.text.startswith("8 Yleiset ehdot\n") for c in chunks)


def test_empty_document_gives_no_chunks():
    assert _chunk([PageText(page=1, text="")]) == []


def test_heading_without_text_becomes_prefix_of_its_subsections():
    pages = [PageText(1, "3 Voimassaolo\n3.2 Vakuutuksen voimassaolo\n3.2.1 Vakuutuspaikassa\n"
                         "Vakuutus on voimassa vakuutuspaikassa.\n3.3 Vakuutetut\nVakuutettuna on ottaja.\n")]
    chunks = _chunk(pages)
    assert [c.section for c in chunks] == ["3.2.1", "3.3"]  # no chunk for the empty 3 and 3.2
    assert chunks[0].text.startswith("3 Voimassaolo\n3.2 Vakuutuksen voimassaolo\n3.2.1 Vakuutuspaikassa\n")
    assert chunks[1].text.startswith("3 Voimassaolo\n3.3 Vakuutetut\n")  # 3.2 is closed by 3.3
