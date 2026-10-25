from app.rag.chunking import _heading, chunk_pages, record_chunk
from app.rag.parsers import PageContent


def test_heading_detection_does_not_split_procedure_steps():
    assert _heading("## 4. Alarm SRVO-050 - Collision detect") == "4. Alarm SRVO-050 - Collision detect"
    assert _heading("3.2 Bell cup cleaning") == "3.2 Bell cup cleaning"
    assert _heading("1. Remove the cover") is None
    assert _heading("2 Remove the four screws holding the cover plate") is None
    assert _heading("DANGER HIGH VOLTAGE") is None
    assert _heading("The alarm SRVO-050 is raised when torque exceeds the limit.") is None


def test_sections_carry_heading_and_identifiers_into_every_chunk():
    body = "\n".join(f"{i}. Step number {i} of the long corrective procedure for the atomizer." for i in range(1, 40))
    text = f"# Alarm manual\n\n## Alarm PNT1-748 - Dump valve leakage\n\nDescription: fluid detected.\n{body}\n\n## Alarm SRVO-050 - Collision\nCheck the path."
    chunks = list(chunk_pages("manual.md", [PageContent(page_num=3, text=text)], "abcd", max_chars=400))
    pnt = [c for c in chunks if c.section_title and "PNT1-748" in c.section_title]
    assert len(pnt) >= 3, "long section must be split"
    assert all("PNT1-748" in c.identifiers for c in pnt), "every chunk inherits the section's alarm code"
    assert all(c.embed_text.startswith("manual.md | ") for c in pnt)
    assert len({c.section_id for c in pnt}) == 1
    srvo = [c for c in chunks if c.section_title and "SRVO-050" in c.section_title]
    assert srvo and srvo[0].section_id != pnt[0].section_id
    assert all(c.page == 3 for c in chunks)
    # Steps stay intact and in order inside chunks.
    assert "1. Step number 1 of" in pnt[0].text


def test_markdown_table_rows_become_record_like_chunks():
    page = PageContent(page_num=1, text="", tables=["| Tag | HiHi |\n| --- | --- |\n| OVN2_Z3_TEMP_PV | 241.3 |"])
    chunks = list(chunk_pages("tags.pdf", [page], "x"))
    assert chunks[0].content_type == "table_row"
    assert chunks[0].text == "Tag: OVN2_Z3_TEMP_PV; HiHi: 241.3"
    assert "OVN2_Z3_TEMP_PV" in chunks[0].identifiers


def test_record_chunk_skips_empty_cells():
    c = record_chunk("wo.csv", "", ["WO", "Alarm", "Note"], ["WO-2021-0000001", "SCD-6841", ""], 7)
    assert c.text == "WO: WO-2021-0000001; Alarm: SCD-6841"
    assert c.page == 7 and c.content_type == "record"
    assert {"WO-2021-0000001", "SCD-6841"} <= set(c.identifiers)
