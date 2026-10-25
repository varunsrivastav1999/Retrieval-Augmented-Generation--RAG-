from app.rag.answer import NOT_FOUND, Source, build_sources, evidence_markdown, verify_answer
from app.rag.search import Hit


def _src(n, text):
    return Source(n=n, chunk_id=n, document_id="d", document="manual.pdf", collection="manuals", page=4,
                  section="Alarm SRVO-050", content_type="text", text=text, snippet=text, score=5.0, matched_ids=[])


SOURCES = [
    _src(1, "Alarm SRVO-050 Collision detect. Inspect the robot path for obstacles. Confirm payload schedule 3."),
    _src(2, "Check conveyor encoder counts against reference marks. Encoder resolution is 2048 pulses."),
]


def test_supported_answer_is_high_confidence():
    answer = ("SRVO-050 means a collision was detected [1]. Inspect the robot path for obstacles [1]. "
              "Check conveyor encoder counts against reference marks [2].")
    v = verify_answer(answer, SOURCES)
    assert v["confidence"] == "high"
    assert v["unknown_identifiers"] == [] and v["unknown_numbers"] == []
    assert v["citations"] == [1, 2]


def test_invented_code_and_value_are_flagged():
    answer = "Replace servo amplifier A06B-6114 [1]. Set torque to 45 Nm [1]. Inspect the robot path [1]."
    v = verify_answer(answer, SOURCES)
    assert "A06B-6114" in v["unknown_identifiers"]
    assert "45" in v["unknown_numbers"]
    assert v["confidence"] == "low"


def test_inferred_section_is_not_counted_as_documented():
    answer = ("Inspect the robot path for obstacles [1].\n\nInferred (not documented)\n"
              "Consider recording vibration trends weekly to catch problems early.")
    assert verify_answer(answer, SOURCES)["confidence"] == "high"


def test_not_found_marker():
    assert verify_answer(NOT_FOUND, SOURCES)["confidence"] == "n/a"


def _hit(i, section, text):
    return Hit(id=i, document_id="d", document="m.pdf", collection="c", file_type="pdf", page=1, section_id=section,
               section_title="S", content_type="text", ord=i, text=text, identifiers=[], score=1.0)


def test_build_sources_dedupes_expanded_sections_and_respects_budget():
    hits = [_hit(1, "s1", "a" * 50), _hit(2, "s1", "b" * 50), _hit(3, "s2", "c" * 300)]
    sources = build_sources(hits, {1: "full section text " * 5}, max_chars=10_000)
    assert [s.chunk_id for s in sources] == [1, 3]   # hit 2 is inside expanded section s1
    assert sources[0].text.startswith("full section text")
    md = evidence_markdown(sources)
    assert "**[1] m.pdf" in md and md.count("> ") >= 2
