import pandas as pd

from rag_receipts.vertex.upload import build_manifest_records, chunk_text_for_upload


def make_metadata_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": f"Attack_range__{i:03d}",
                "page_title": "Attack range",
                "url": "https://oldschool.runescape.wiki/w/Attack_range",
                "section_path": "Overview",
                "heading_anchor": "Overview",
                "token_count": 100 + i,
                "source_type": "prose",
                "text": f"chunk text {i}",
            }
            for i in range(n)
        ]
    )


def test_chunk_text_for_upload_matches_breadcrumb_form():
    text = chunk_text_for_upload("Attack range", "Overview", "Some body text.")

    assert text == "Attack range > Overview\n\nSome body text."


def test_chunk_text_for_upload_no_section_path():
    text = chunk_text_for_upload("Attack range", "", "Some body text.")

    assert text == "Attack range\n\nSome body text."


def test_build_manifest_records_one_per_row():
    metadata = make_metadata_df(3)

    records = build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")

    assert len(records) == 3


def test_build_manifest_records_fields():
    metadata = make_metadata_df(1)

    [record] = build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")

    assert record["id"] == "Attack_range__000"
    assert record["structData"]["chunk_id"] == "Attack_range__000"
    assert record["structData"]["page_title"] == "Attack range"
    assert record["structData"]["url"] == "https://oldschool.runescape.wiki/w/Attack_range"
    assert record["structData"]["section_path"] == "Overview"
    assert record["structData"]["heading_anchor"] == "Overview"
    assert record["structData"]["source_type"] == "prose"
    assert record["structData"]["token_count"] == 100
    assert record["content"] == {
        "mimeType": "text/plain",
        "uri": "gs://my-bucket/corpus/chunks/Attack_range__000.txt",
    }


def test_build_manifest_records_chunk_id_survives_as_both_id_and_struct_data():
    metadata = make_metadata_df(1)

    [record] = build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")

    assert record["id"] == record["structData"]["chunk_id"]
