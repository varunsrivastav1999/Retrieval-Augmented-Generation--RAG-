from app.rag.answer import Source, exact_answer, exact_markdown
from app.rag.intent import detect_intent

PROC = """Interval: every 2000 operating hours. Required parts: O-ring kit P/N 8301-5662-59.
CAUTION: Wear solvent-resistant gloves and safety glasses.
1. Move the robot to the maintenance position and switch the controller to T1 mode.
2. Close the paint supply valve and purge the bell cup with solvent for 30 s.
3. Remove the four M5 screws that hold the bell
cup cover.
4. Tighten the M5 screws crosswise to 8 Nm."""

ROW = ("Code: ATM-174; Message: Air turbine communication timeout; Probable cause: Fieldbus connector is loose.; "
       "Remedy: Replace the air turbine using the procedure in chapter 3 and restart the robot.")


def src(text, section="3.1 Replacing the bell cup"):
    return Source(n=1, chunk_id=1, document_id="d", document="p-500_maintenance_manual.pdf", collection="c", page=5,
                  section=section, content_type="text", text=text, snippet=text, score=1.0, matched_ids=[])


def ask(q, text, **kw):
    return exact_answer(q, detect_intent(q), [src(text, **kw)], [])


def test_full_procedure_is_verbatim_and_ordered():
    q = ask("Procedure for replacing the bell cup on the P-500", PROC)[0]["text"]
    assert "3. Remove the four M5 screws that hold the bell cup cover." in q   # wrapped line re-joined
    assert q.index("1. Move") < q.index("2. Close") < q.index("4. Tighten")
    assert "CAUTION: Wear solvent-resistant gloves and safety glasses." in q


def test_single_value_questions_quote_the_exact_line():
    assert ask("What torque do I use for the bell cup screws?", PROC)[0]["text"] == "4. Tighten the M5 screws crosswise to 8 Nm."
    assert "P/N 8301-5662-59" in ask("Which O-ring kit part number is needed?", PROC)[0]["text"]
    assert ask("What is step 2?", PROC)[0]["text"].startswith("2. Close the paint supply valve")


def test_table_row_is_quoted_whole_with_its_code():
    out = ask("What is the remedy for alarm ATM-174?", ROW, section="2.1 Alarm list")[0]["text"]
    assert out == ROW  # code and remedy stay together, word for word
    assert "Remedy: Replace the air turbine using the procedure in chapter 3 and restart the robot." in out


def test_markdown_cites_document_and_page():
    md = exact_markdown(ask("What torque?", PROC))
    assert "> 4. Tighten the M5 screws crosswise to 8 Nm." in md and "page 5" in md and "p-500_maintenance_manual.pdf" in md


def test_when_in_a_procedure_question_is_not_an_alarm_condition_cue():
    text = PROC + "\nCondition: The measured speed differs from the commanded speed by more than 5 %."
    out = ask("What torque do I use when replacing the bell cup?", text)[0]["text"]
    assert out == "4. Tighten the M5 screws crosswise to 8 Nm."


def test_two_part_question_gets_a_quote_for_each_part():
    row = src(ROW, section="2.1 Alarm list")
    detail = Source(n=2, chunk_id=2, document_id="d", document="p-500_maintenance_manual.pdf", collection="c", page=4,
                    section="2.2 Alarm ATM-174 Air turbine communication timeout", content_type="text",
                    text="Condition: No valid telegram was received within 100 ms.\nProbable cause: Fieldbus connector is loose.",
                    snippet="", score=1.0, matched_ids=[])
    q = "When is alarm ATM-174 raised and what is the remedy?"
    out = exact_answer(q, detect_intent(q), [row, detail], ["ATM-174"])
    joined = "\n".join(x["text"] for x in out)
    assert "Remedy: Replace the air turbine" in joined and "Condition: No valid telegram" in joined
    assert len(out) == 2
