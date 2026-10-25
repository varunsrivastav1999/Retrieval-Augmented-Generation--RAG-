from app.rag.identifiers import extract_identifiers, normalize_identifier


def test_alarm_codes_tags_addresses_work_orders():
    text = "Robot R07 stopped with PNT1-128 / SRVO-050; tag OVN2_Z3_TEMP_PV at DB372.DBD520 (WO-2024-0013245)"
    ids = extract_identifiers(text)
    for expected in ["R07", "PNT1-128", "SRVO-050", "OVN2_Z3_TEMP_PV", "DB372.DBD520", "WO-2024-0013245"]:
        assert expected in ids


def test_plain_numbers_dates_and_words_are_not_identifiers():
    assert extract_identifiers("Set 23 C at 6 bar on 2024-03-14 version 2.5.1, then wait 10:30") == []
    assert extract_identifiers("paint is sagging on vertical panels") == []


def test_other_plc_styles():
    ids = extract_identifiers("Check %MW100, N7:0 and BOOTH1.EXH_FAN.MTR_TEMP with E-102 / F0021")
    assert {"%MW100", "N7:0", "BOOTH1.EXH_FAN.MTR_TEMP", "E-102", "F0021"} <= set(ids)


def test_normalize_is_case_insensitive():
    assert normalize_identifier(" srvo-050 ") == "SRVO-050"
