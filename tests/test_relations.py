from app.rag.relations import _kind, _label


def test_labels_only_come_from_what_the_document_states():
    assert _label("ATM-619", "2.3 Alarm ATM-619 Shaping air ring leak detected", "") == "Shaping air ring leak detected"
    row = "Code: ATM-619; Message: Shaping air ring leak detected; Probable cause: Valve seal is worn."
    assert _label("ATM-619", "2.1 Alarm list", row) == "Shaping air ring leak detected"
    assert _label("R16", "Alarm reference", "Typically reported on robots R10, R21, R16. joint limit") == ""


def test_kinds():
    assert [_kind(i) for i in ["SRVO-822", "R16", "WO-2021-0000001", "8301-5662-59", "DB310.DBD476", "PK1_Z6_VIB_PV"]] == \
        ["code", "robot", "work_order", "part", "address", "tag"]
