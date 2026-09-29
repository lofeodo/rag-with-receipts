from conftest import load_fixture

from rag_receipts.ingestion.html_parser import build_section_tree, flatten_infobox, flatten_table
from bs4 import BeautifulSoup


def test_boilerplate_stripped_prose_preserved():
    parsed = build_section_tree(load_fixture("basic_page.html"), "Basic Page")
    assert len(parsed.sections) == 1
    section = parsed.sections[0]
    assert section.heading_path == ["Overview"]
    text = " ".join(section.blocks)
    assert "first paragraph" in text
    assert "second paragraph" in text
    assert "Should not appear" not in text
    assert "edit" not in text


def test_nested_heading_paths_and_parent_intro_captured():
    parsed = build_section_tree(load_fixture("nested_headings.html"), "Nested Page")
    by_path = {" > ".join(s.heading_path): s for s in parsed.sections}

    assert "Combat styles" in by_path
    assert "intro paragraph" in " ".join(by_path["Combat styles"].blocks)

    assert "Combat styles > Melee" in by_path
    assert "close-range" in " ".join(by_path["Combat styles > Melee"].blocks)

    assert "Combat styles > Ranged" in by_path
    assert "distance-based" in " ".join(by_path["Combat styles > Ranged"].blocks)

    assert "Strategy" in by_path
    assert "sibling" in " ".join(by_path["Strategy"].blocks)


def test_flatten_infobox_key_value_lines():
    soup = BeautifulSoup(load_fixture("infobox_page.html"), "lxml")
    lines = flatten_infobox(soup)
    assert lines is not None
    assert "Released: 1 January 2020" in lines
    assert any(line.startswith("Examine:") and "sharp tooth" in line for line in lines)
    assert any(line.startswith("Combat level:") for line in lines)
    # structural rows (header, padding) must not produce spurious entries
    assert not any("Test Monster" in line for line in lines)


def test_infobox_page_has_dedicated_infobox_section_and_prose_section():
    parsed = build_section_tree(load_fixture("infobox_page.html"), "Infobox Page")
    infobox_sections = [s for s in parsed.sections if s.source_type == "infobox"]
    assert len(infobox_sections) == 1
    assert infobox_sections[0].heading_anchor == "infobox"

    prose_sections = [s for s in parsed.sections if s.heading_path == ["Overview"]]
    assert len(prose_sections) == 1
    assert "normal prose section" in " ".join(prose_sections[0].blocks)


def test_flatten_table_header_row_and_data_rows():
    soup = BeautifulSoup(load_fixture("table_page.html"), "lxml")
    table = soup.select_one("table.wikitable")
    rows = flatten_table(table)
    assert rows[0] == "Level: 1 | XP: 25 | Action: Chop logs"
    assert rows[-1] == "Level: 30 | XP: 60 | Action: Chop willow logs"


def test_table_heavy_section_classified_as_table_source_type():
    parsed = build_section_tree(load_fixture("table_page.html"), "Table Page")
    assert len(parsed.sections) == 1
    assert parsed.sections[0].source_type == "table"
    assert parsed.sections[0].heading_path == ["Training methods"]


def test_no_boilerplate_leakage():
    parsed = build_section_tree(load_fixture("boilerplate_page.html"), "Boilerplate Page")
    all_text = " ".join(block for s in parsed.sections for block in s.blocks)
    assert "Genuine content" in all_text
    for leaked in ["Retrieved from", "Categories:", "reference text"]:
        assert leaked not in all_text
