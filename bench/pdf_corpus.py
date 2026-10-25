"""
Realistic PDF / DOCX maintenance library + verbatim-answer question set.

Each manual looks like a real OEM document: running header/footer on every page,
numbered chapters (1 Safety, 2 Alarm reference, 3 Maintenance procedures,
4 Parameters, 5 Troubleshooting), numbered steps with WARNING/CAUTION notes,
torque values and part numbers, an alarm table that continues across pages,
parameter tables and a troubleshooting matrix. One manual page is rasterised
("scanned") so it can only be read through OCR.

Every question records the EXACT sentence/step/table cell that must appear
word for word in the answer, and the page it is on.

  python bench/pdf_corpus.py --out bench/data_pdf --manuals 24
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

SEED = 7
STYLES = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=STYLES["Heading1"], fontSize=16, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=STYLES["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("b", parent=STYLES["BodyText"], fontSize=9.5, leading=12.5)
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.5, leading=10.5)
WARN = ParagraphStyle("w", parent=BODY, textColor=colors.HexColor("#8a1c1c"))

MODELS = ["P-500", "P-520", "P-700", "P-720", "SL-300", "SL-350", "BX-90", "BX-120", "EcoBell-3", "EcoBell-4",
          "RX-15", "RX-30"]
SUBSYSTEMS = {
    "bell atomizer": ["bell cup", "air turbine", "shaping air ring", "HV cascade", "paint valve", "dump valve"],
    "wrist": ["J4 gearbox", "J5 gearbox", "J6 gearbox", "wrist hose set", "wrist cover"],
    "color changer": ["color valve", "solvent valve", "air purge valve", "manifold seal"],
    "paint supply": ["gear pump", "pressure regulator", "flow meter", "filter element"],
    "controller": ["servo amplifier", "safety relay", "fieldbus coupler", "cooling fan", "backup battery"],
}
FAULTS = [
    ("speed deviation", "The measured speed differs from the commanded speed by more than 5 % for longer than 2 s.",
     ["Bearing air pressure below 5.5 bar.", "Speed sensor fibre is dirty or damaged.", "Turbine bearing is worn."]),
    ("overtemperature", "The temperature sensor reports more than 70 °C for longer than 10 s.",
     ["Cooling air filter is clogged.", "Cooling fan has stopped.", "Ambient temperature in the cabinet exceeds 40 °C."]),
    ("pressure low", "The measured pressure stays below the configured minimum for longer than 3 s.",
     ["Supply filter element is blocked.", "A hose or fitting is leaking.", "The pump is worn."]),
    ("communication timeout", "No valid telegram was received from the device within the watchdog time of 100 ms.",
     ["Fieldbus connector is loose.", "Cable shield is not grounded.", "Device address is duplicated."]),
    ("leak detected", "The leak sensor in the housing has detected liquid.",
     ["Valve seal is worn.", "Fitting is loose.", "Paint line is cracked."]),
    ("high voltage fault", "The high voltage dropped below 40 kV or the current limit of 150 µA was exceeded.",
     ["Paint deposits on the shroud.", "Conductive paint short-circuit to ground.", "Faulty HV cascade."]),
]


def build_facts(rng: random.Random, n_manuals: int):
    manuals = []
    used_codes = set()
    for m in range(n_manuals):
        model = MODELS[m % len(MODELS)] + ("" if m < len(MODELS) else f"-{m // len(MODELS) + 1}")
        sub = list(SUBSYSTEMS)[m % len(SUBSYSTEMS)]
        alarms = []
        for _ in range(rng.randint(14, 22)):
            comp = rng.choice(SUBSYSTEMS[sub])
            fault, cond, causes = rng.choice(FAULTS)
            while True:
                code = f"{rng.choice(['ATM', 'WRS', 'CCV', 'PSU', 'CTL'])}-{rng.randint(100, 999)}"
                if code not in used_codes:
                    used_codes.add(code)
                    break
            remedy = rng.choice([
                f"Clean or replace the {comp} and reset the alarm on the teach pendant.",
                f"Check the {comp} connections, then acknowledge the alarm with the RESET key.",
                f"Replace the {comp} using the procedure in chapter 3 and restart the robot.",
            ])
            alarms.append({"code": code, "message": f"{comp.capitalize()} {fault}", "condition": cond,
                           "cause": rng.choice(causes), "remedy": remedy, "comp": comp})
        procedures = []
        for k, comp in enumerate(SUBSYSTEMS[sub][:4], 1):
            torque = rng.choice([4.5, 6, 8, 12, 18, 25])
            pn = f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"
            interval = rng.choice(["every 500 operating hours", "every 3 months", "every 2000 operating hours",
                                   "every week"])
            steps = [
                "Move the robot to the maintenance position and switch the controller to T1 mode.",
                f"Close the paint supply valve and purge the {comp} with solvent for 30 s.",
                f"Remove the four M5 screws that hold the {comp} cover.",
                f"Remove the {comp} and inspect the sealing surfaces for scratches.",
                f"Install the new O-ring kit P/N {pn} on the {comp}.",
                f"Tighten the M5 screws crosswise to {torque} Nm.",
                "Run the leak test for 10 minutes at 6 bar and record the result in the maintenance log.",
            ]
            procedures.append({"title": f"Replacing the {comp}", "comp": comp, "steps": steps, "pn": pn,
                               "torque": torque, "interval": interval, "no": k})
        params = []
        for p in range(rng.randint(10, 16)):
            name = f"{sub[:3].upper()}_{rng.choice(['SPD', 'PRS', 'TMP', 'FLW', 'KV'])}_{p:02d}"
            lo = rng.randint(0, 50)
            params.append({"name": name, "default": lo + rng.randint(5, 50), "min": lo, "max": lo + 200,
                           "unit": rng.choice(["rpm", "bar", "°C", "cc/min", "kV"])})
        manuals.append({"model": model, "subsystem": sub, "alarms": alarms, "procedures": procedures,
                        "params": params, "rev": rng.choice("ABCDE")})
    return manuals


def _header_footer(model, rev):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, 287 * mm, f"{model} Painting Robot - Maintenance Manual - Rev {rev}")
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
        canvas.drawString(18 * mm, 10 * mm, "Confidential - for trained maintenance personnel only")
        canvas.restoreState()
    return draw


def _table(rows, widths):
    t = Table([[Paragraph(str(c), CELL) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e9ef")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _scanned_image(lines):
    """Rasterise a block of text (simulates a scanned page that only OCR can read)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    img = PILImage.new("L", (1700, 90 + 60 * len(lines)), 255)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        d.text((40, 40 + i * 60), line, fill=0, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image(buf, width=170 * mm, height=170 * mm * img.height / img.width)


def write_pdf(path, man, scanned=False):
    story = [Paragraph(f"{man['model']} Painting Robot", H1),
             Paragraph(f"Maintenance Manual - {man['subsystem'].title()} - Revision {man['rev']}", BODY),
             Spacer(1, 6), PageBreak()]
    story += [Paragraph("1 Safety", H1),
              Paragraph("1.1 General safety instructions", H2),
              Paragraph("Only trained and authorised personnel may carry out the work described in this manual. "
                        "Before any work inside the booth, switch off the high voltage, lock out the main switch "
                        "and wait at least 5 minutes for the HV cascade to discharge.", BODY),
              Paragraph("WARNING: Solvent vapours are flammable. Never use compressed air to dry the bell cup.", WARN),
              PageBreak()]
    story += [Paragraph("2 Alarm reference", H1),
              Paragraph("2.1 Alarm list", H2),
              Paragraph("The following table lists all alarms of this subsystem with cause and remedy.", BODY)]
    rows = [["Code", "Message", "Probable cause", "Remedy"]]
    rows += [[a["code"], a["message"], a["cause"], a["remedy"]] for a in man["alarms"]]
    story.append(_table(rows, [20 * mm, 40 * mm, 50 * mm, 64 * mm]))
    for i, a in enumerate(man["alarms"][:8], 2):
        story += [Paragraph(f"2.{i} Alarm {a['code']} {a['message']}", H2),
                  Paragraph(f"Condition: {a['condition']}", BODY),
                  Paragraph(f"Probable cause: {a['cause']}", BODY),
                  Paragraph(f"Remedy: {a['remedy']}", BODY)]
    story.append(PageBreak())
    story.append(Paragraph("3 Maintenance procedures", H1))
    for p in man["procedures"]:
        block = [Paragraph(f"3.{p['no']} {p['title']}", H2),
                 Paragraph(f"Interval: {p['interval']}. Required parts: O-ring kit P/N {p['pn']}.", BODY),
                 Paragraph("CAUTION: Wear solvent-resistant gloves and safety glasses.", WARN)]
        block += [Paragraph(f"{n}. {s}", BODY) for n, s in enumerate(p["steps"], 1)]
        story.append(KeepTogether(block))
    story.append(PageBreak())
    story += [Paragraph("4 Parameters", H1), Paragraph("4.1 Parameter list", H2)]
    prow = [["Parameter", "Default", "Minimum", "Maximum", "Unit"]]
    prow += [[p["name"], p["default"], p["min"], p["max"], p["unit"]] for p in man["params"]]
    story.append(_table(prow, [45 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm]))
    story += [PageBreak(), Paragraph("5 Troubleshooting", H1), Paragraph("5.1 Troubleshooting matrix", H2)]
    trows = [["Symptom", "Possible cause", "Check"]]
    for a in man["alarms"][8:14]:
        trows.append([f"{a['message']} ({a['code']})", a["cause"], a["remedy"]])
    story.append(_table(trows, [55 * mm, 55 * mm, 64 * mm]))
    if scanned:
        a = man["alarms"][-1]
        story += [PageBreak(), Paragraph("6 Service bulletin (scanned)", H1),
                  _scanned_image([f"SERVICE BULLETIN SB-{man['model']}-01",
                                  f"Alarm {a['code']}: {a['message']}.",
                                  "Replace the filter element every 250 operating hours.",
                                  "Use only original spare parts."])]
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm, title=f"{man['model']} maintenance manual")
    hf = _header_footer(man["model"], man["rev"])
    doc.build(story, onFirstPage=hf, onLaterPages=hf)


def write_docx(path, man):
    import docx
    d = docx.Document()
    d.add_heading(f"{man['model']} - Work instructions", level=1)
    for p in man["procedures"][:2]:
        d.add_heading(f"WI-{p['no']} {p['title']} (work instruction)", level=2)
        d.add_paragraph(f"Interval: {p['interval']}.")
        for n, s in enumerate(p["steps"], 1):
            d.add_paragraph(f"{n}. {s}")
    t = d.add_table(rows=1, cols=3)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text, t.rows[0].cells[2].text = "Part", "P/N", "Torque"
    for p in man["procedures"]:
        r = t.add_row().cells
        r[0].text, r[1].text, r[2].text = p["comp"], p["pn"], f"{p['torque']} Nm"
    d.save(path)


def questions(rng, manuals):
    qs = []
    for man in manuals:
        f = f"{man['model'].lower()}_maintenance_manual.pdf"
        for a in rng.sample(man["alarms"], 3):
            qs.append({"category": "alarm_remedy", "query": f"What is the remedy for alarm {a['code']}?",
                       "verbatim": a["remedy"], "doc": f})
        a = rng.choice(man["alarms"][:8])
        qs.append({"category": "alarm_condition", "query": f"When is alarm {a['code']} raised?",
                   "verbatim": a["condition"], "doc": f})
        p = rng.choice(man["procedures"])
        step_no = rng.choice([3, 5, 6])
        qs.append({"category": "procedure_step",
                   "query": f"On the {man['model']}, what torque do I use when replacing the {p['comp']}?",
                   "verbatim": p["steps"][5], "doc": f,
                   # the work-instruction table states the same value word for word
                   "alternatives": [f"Torque: {p['torque']} Nm"]})
        qs.append({"category": "procedure_full",
                   "query": f"{man['model']}: procedure for replacing the {p['comp']}",
                   "verbatim": p["steps"][step_no - 1], "doc": f, "all_steps": p["steps"]})
        qs.append({"category": "part_number", "query": f"Which O-ring kit part number is needed for the {p['comp']} on the {man['model']}?",
                   "verbatim": p["pn"], "doc": f})
        pr = rng.choice(man["params"])
        qs.append({"category": "parameter", "query": f"What is the default value and range of parameter {pr['name']} on the {man['model']}?",
                   "verbatim": str(pr["default"]), "doc": f, "also": [pr["name"], str(pr["max"])]})
    for man in manuals[:6]:
        a = man["alarms"][-1]
        qs.append({"category": "scanned_ocr", "query": f"What does service bulletin SB-{man['model']}-01 say about alarm {a['code']}?",
                   "verbatim": "Replace the filter element every 250 operating hours.", "doc": f"{man['model'].lower()}_maintenance_manual.pdf"})
    for man in manuals[:8]:
        p = man["procedures"][0]
        qs.append({"category": "docx_instruction", "query": f"Work instruction WI-1 for the {man['model']}: what is step 2?",
                   "verbatim": p["steps"][1], "doc": f"{man['model'].lower()}_work_instructions.docx"})
    for neg in ["What is the torque for the spot welding gun electrode cap?",
                "How do I replace the forklift hydraulic pump?",
                "What is alarm XYZ-999 on the stamping press?",
                "What is the coolant concentration for the CNC lathe?"]:
        qs.append({"category": "negative", "query": neg, "verbatim": None})
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/data_pdf")
    ap.add_argument("--manuals", type=int, default=24)
    args = ap.parse_args()
    rng = random.Random(SEED)
    os.makedirs(args.out, exist_ok=True)
    manuals = build_facts(rng, args.manuals)
    for i, man in enumerate(manuals):
        write_pdf(os.path.join(args.out, f"{man['model'].lower()}_maintenance_manual.pdf"), man, scanned=i < 6)
        if i < 8:
            write_docx(os.path.join(args.out, f"{man['model'].lower()}_work_instructions.docx"), man)
    qs = questions(rng, manuals)
    with open(os.path.join(args.out, "questions.jsonl"), "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    print(f"{len(manuals)} PDF manuals, 8 DOCX, {len(qs)} questions -> {args.out}")


if __name__ == "__main__":
    main()
