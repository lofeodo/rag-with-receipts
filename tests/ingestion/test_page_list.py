from rag_receipts.ingestion.page_list import extract_internal_links
from rag_receipts.ingestion.utils import breadcrumb, make_chunk_id, page_url, slugify


def test_slugify_replaces_spaces_and_slashes():
    assert slugify("Monkey Madness I/Quick guide") == "Monkey_Madness_I_Quick_guide"


def test_slugify_strips_invalid_characters():
    assert slugify("Duradel and assignment page") == "Duradel_and_assignment_page"
    assert slugify("Dragon Slayer I's") == "Dragon_Slayer_I_s"


def test_page_url_builds_canonical_link():
    assert page_url("Dragon Slayer I") == "https://oldschool.runescape.wiki/w/Dragon_Slayer_I"


def test_breadcrumb_with_and_without_section():
    assert breadcrumb("Slayer training", "Methods > Early game") == "Slayer training > Methods > Early game"
    assert breadcrumb("Slayer training", "") == "Slayer training"


def test_make_chunk_id_is_zero_padded_and_sequential():
    assert make_chunk_id("Slayer_training", 0) == "Slayer_training__000"
    assert make_chunk_id("Slayer_training", 12) == "Slayer_training__012"


HTML_WITH_LINKS = """
<div class="mw-parser-output">
  <p>See <a href="/w/Abyssal_demon" title="Abyssal demon">Abyssal demon</a> and
  <a href="/w/Dragon_dagger#Poisoned" title="Dragon dagger">Dragon dagger</a>.</p>
  <p>Ignore <a href="/w/Category:Monsters">Category:Monsters</a>,
  <a href="/w/File:Abyssal_demon.png">a file</a>, and
  <a href="/w/Special:RecentChanges">special pages</a>.</p>
</div>
<div class="navbox">
  <a href="/w/Should_not_appear">Should not appear</a>
</div>
"""


def test_extract_internal_links_filters_namespaces_and_navbox():
    links = extract_internal_links(HTML_WITH_LINKS)
    assert links == ["Abyssal demon", "Dragon dagger"]
