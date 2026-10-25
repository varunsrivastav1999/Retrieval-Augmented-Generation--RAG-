import pytest

from app.rag.intent import detect_intent, is_greeting


@pytest.mark.parametrize("query,expected", [
    ("What does alarm SRVO-050 mean?", "alarm"),
    ("PNT1-128 how do I reset it", "alarm"),
    ("How do I replace the bell cup?", "procedure"),
    ("What is the monthly preventive maintenance for the conveyor?", "procedure"),
    ("Paint is sagging on vertical panels, what should I check?", "troubleshoot"),
    ("What is the HiHi limit of OVN2_Z3_TEMP_PV?", "parameter"),
    ("Previous work orders for R07 turbine bearing", "history"),
    ("Compare SRVO-050 and SRVO-062", "comparison"),
])
def test_intents(query, expected):
    assert detect_intent(query).name == expected


def test_greeting():
    assert is_greeting("hello!")
    assert not is_greeting("hello, what does SRVO-050 mean?")
