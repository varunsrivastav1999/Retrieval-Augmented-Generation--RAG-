"""
Deterministic synthetic corpus for robot-painting maintenance benchmarks.

Generates realistic-looking (but fictional) maintenance documents:
  * alarm_manual_*.md     robot / PLC / SCADA alarm catalogue (one section per alarm)
  * tag_list.csv          SCADA / PLC tag dictionary (one row per tag)
  * process_defects.md    painting-process defect troubleshooting guide
  * pm_procedures.md      preventive maintenance procedures
  * work_orders.csv       historical maintenance cases

and a labelled query set (queries.jsonl). A query is answered correctly when a
retrieved chunk contains the query's `expect` marker string (an alarm code,
tag name, work-order id, ...). Negative queries have expect=null and should be
refused.

Usage:
  python bench/corpus.py --out bench/data --alarms 600 --tags 1500 --work-orders 8000
  python bench/corpus.py --out /tmp/big --work-orders 1000000 --no-queries   # scale corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

SEED = 20261025

ROBOTS = [f"R{n:02d}" for n in range(1, 25)]
ZONES = ["Prime Booth 1", "Prime Booth 2", "Base Coat Booth", "Clear Coat Booth", "Sealer Station", "Spot Repair Booth"]

# (subsystem, alarm prefix, component nouns)
SUBSYSTEMS = [
    ("servo", "SRVO", ["J1 axis motor", "J2 axis motor", "J3 axis motor", "J4 wrist motor", "J5 wrist motor", "J6 wrist motor", "servo amplifier", "pulse coder"]),
    ("atomizer", "PNT1", ["bell cup", "air turbine", "shaping air valve", "high-voltage cascade", "fluid regulator", "color change valve", "dump valve", "trigger valve"]),
    ("motion", "MOTN", ["path planner", "joint limit", "conveyor tracking encoder", "collision guard", "TCP frame", "interference zone"]),
    ("plc", "PLC", ["booth exhaust fan VFD", "supply air house damper", "oven burner management relay", "conveyor drive", "pretreatment pump", "ED rectifier", "sludge conveyor", "safety PLC"]),
    ("scada", "SCD", ["historian link", "OPC UA server", "alarm server", "HMI node", "tag database", "redundant server pair"]),
    ("process", "PRC", ["booth humidity controller", "booth temperature controller", "paint kitchen circulation pump", "viscosity sensor", "film build gauge", "oven zone thermocouple"]),
]

FAILURE_MODES = [
    ("overcurrent", "Current drawn exceeded the configured trip limit", ["mechanical binding or seized bearing", "short circuit in motor cable", "incorrect drive parameter set"], ["Check the motor and gearbox for mechanical binding by rotating by hand with brakes released.", "Megger-test the motor power cable between drive and motor (>100 MOhm expected).", "Compare drive parameter set against the commissioning backup and restore if different."]),
    ("overtemperature", "Measured temperature rose above the high-high limit", ["blocked cooling air path", "failed cooling fan", "ambient temperature too high"], ["Clean the cooling fins and intake filter.", "Verify the cooling fan rotates and draws nominal current.", "Measure cabinet ambient temperature; it must stay below 40 C."]),
    ("communication loss", "Cyclic communication with the device timed out", ["loose or damaged network connector", "faulty switch port", "IP address conflict"], ["Reseat the network connector at both ends and inspect for bent pins.", "Move the cable to a spare switch port and check link LEDs.", "Run an address scan to confirm no duplicate IP address on the segment."]),
    ("speed deviation", "Actual speed deviated from the setpoint beyond tolerance", ["worn turbine or motor bearing", "low supply air pressure", "contaminated speed feedback sensor"], ["Measure supply air pressure at the regulator; nominal is 6 bar.", "Inspect bearing condition and replace if noise or play is detected.", "Clean the speed feedback fiber optic sensor and re-check the signal level."]),
    ("pressure low", "Pressure fell below the low alarm setpoint", ["clogged filter element", "leaking hose or fitting", "pump wear"], ["Replace the inline filter element and record the differential pressure.", "Leak-test hoses and fittings with the system pressurised.", "Check pump discharge pressure against the pump curve."]),
    ("position deviation", "Following error exceeded the allowed window", ["loose coupling", "encoder contamination", "excessive payload or acceleration"], ["Check coupling set screws for torque marks and re-torque to specification.", "Inspect the encoder for paint contamination and clean.", "Verify payload settings and reduce acceleration override."]),
    ("leakage detected", "Fluid was detected in the leak sensor housing", ["worn valve seal", "cracked fluid line", "loose fitting"], ["Isolate fluid supply and depressurise the line.", "Replace the valve seal kit.", "Re-torque fittings and run a 10 minute leak check."]),
    ("high voltage fault", "Electrostatic high voltage dropped or current limit tripped", ["paint build-up on atomizer shroud", "conductive paint short to ground", "faulty cascade module"], ["Clean the atomizer shroud and bell cup with approved solvent.", "Check the paint resistivity; it must be above 0.5 MOhm-cm for direct charge.", "Measure cascade output with the HV probe and replace the cascade if below 90 kV at no load."]),
    ("collision detect", "Unexpected disturbance torque indicated a collision", ["obstacle in robot path", "incorrect payload data", "conveyor tracking error"], ["Inspect the robot path and booth for obstacles or fallen parts.", "Confirm payload schedule matches the installed atomizer.", "Check conveyor encoder counts against reference marks."]),
    ("watchdog timeout", "Controller watchdog expired before the task cycle completed", ["CPU overload", "corrupted program block", "firmware mismatch"], ["Check CPU load diagnostics and reduce scan-time critical logic.", "Download the verified program backup.", "Compare firmware version with the approved baseline."]),
]

DEFECTS = [
    ("Sags and runs", "Paint flows downward on vertical panels forming curtains or drips", ["film build too high", "paint viscosity too low", "booth temperature too low slowing flash-off"], ["Reduce fluid flow rate by 10 percent", "Increase viscosity to specification", "Raise booth temperature to 23 C"], ["fluid flow", "viscosity", "booth temperature"]),
    ("Orange peel", "Surface shows a dimpled texture similar to orange skin", ["bell speed too low causing coarse atomization", "shaping air too high", "paint viscosity too high"], ["Increase bell speed to 45,000 rpm", "Reduce shaping air", "Thin the paint to target viscosity"], ["bell speed", "shaping air", "viscosity"]),
    ("Craters", "Small round depressions with a raised rim in the paint film", ["silicone or oil contamination on the substrate", "contaminated compressed air", "incompatible sealer"], ["Wipe parts with approved cleaner", "Check compressed air oil content and coalescing filters", "Audit sealer batch compatibility"], ["compressed air quality", "substrate cleanliness"]),
    ("Dirt inclusions", "Particles embedded in the cured paint film", ["booth air filter breakthrough", "dirty tack-off process", "operator clothing fibers"], ["Replace ceiling filters", "Inspect feather duster and tack-off station", "Enforce lint-free coveralls"], ["booth air balance", "filter differential pressure"]),
    ("Low film build", "Dry film thickness below the minimum specification", ["fluid flow too low", "transfer efficiency drop from low high voltage", "excessive bell-to-target distance"], ["Increase fluid flow", "Verify high voltage set to 70 kV", "Adjust robot path distance to 200 mm"], ["fluid flow", "high voltage", "target distance"]),
    ("Solvent popping", "Small blisters or pinholes caused by trapped solvent", ["flash-off time too short", "oven ramp too fast", "film build too high"], ["Extend flash-off by 2 minutes", "Reduce oven zone 1 setpoint", "Reduce film build"], ["flash time", "oven profile"]),
    ("Color mismatch", "Panel color differs from standard beyond delta E tolerance", ["incomplete color change purge", "wrong paint batch", "film build variation"], ["Extend solvent purge sequence", "Verify paint batch against the color standard", "Check film build uniformity"], ["color change", "film build"]),
    ("Overspray haze", "Dull mist of dry paint particles on adjacent surfaces", ["booth downdraft velocity too low", "bell-to-target distance too large", "shaping air too low"], ["Balance booth downdraft to 0.4 m/s", "Reduce target distance", "Increase shaping air"], ["downdraft velocity", "shaping air"]),
    ("Mottling", "Uneven blotchy appearance in metallic base coat", ["uneven film build", "wet-on-wet application timing", "incorrect flake orientation from low bell speed"], ["Adjust overlap of robot passes", "Adjust flash time between coats", "Increase bell speed"], ["bell speed", "overlap", "flash time"]),
    ("Poor adhesion", "Paint film flakes or fails cross-hatch tape test", ["pretreatment phosphate coating weight low", "contaminated substrate", "under-cure in oven"], ["Check pretreatment bath chemistry", "Verify rinse conductivity", "Verify oven cure window with data logger"], ["pretreatment chemistry", "oven cure"]),
]

DEFECT_PARAPHRASES = {
    "Sags and runs": "paint is dripping and forming curtains on the doors",
    "Orange peel": "the finish looks bumpy like the skin of a citrus fruit",
    "Craters": "small round holes with raised edges appear in the coating",
    "Dirt inclusions": "specks of dust are trapped inside the cured paint",
    "Low film build": "measured dry coating thickness is under spec",
    "Solvent popping": "tiny blisters and pinholes appear after the oven",
    "Color mismatch": "panel shade does not match the colour standard",
    "Overspray haze": "dull dry mist of paint lands on nearby surfaces",
    "Mottling": "metallic basecoat looks blotchy and uneven",
    "Poor adhesion": "paint flakes off during the cross hatch tape test",
}

PM_EQUIPMENT = [
    ("Bell atomizer", ["Remove and clean the bell cup with solvent", "Inspect bell cup edge for nicks under magnification", "Check turbine bearing air consumption", "Replace shaping air ring O-rings"]),
    ("Paint robot wrist", ["Check grease level in J4-J6 gearboxes", "Inspect wrist hoses for chafing", "Verify wrist covers are intact", "Record backlash measurement"]),
    ("Booth exhaust fan", ["Check belt tension and alignment", "Grease fan bearings", "Measure vibration at both bearings", "Inspect VFD cooling fan"]),
    ("Oven burner", ["Inspect flame sensor and clean lens", "Verify combustion air switch operation", "Check gas valve leak test results", "Record zone temperature uniformity"]),
    ("Paint kitchen", ["Clean circulation pump strainers", "Calibrate viscosity sensor", "Inspect agitator seals", "Check back-pressure regulator setpoints"]),
    ("Conveyor", ["Lubricate chain and trolleys", "Check take-up travel", "Inspect drive gearbox oil level", "Verify encoder mounting"]),
    ("PLC panel", ["Back up PLC program and compare checksum", "Check battery status", "Inspect terminal tightness with torque driver", "Clean panel filters"]),
    ("SCADA server", ["Verify historian archive backup", "Check disk free space above 20 percent", "Review alarm flood statistics", "Test redundant server failover"]),
]

MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def build_alarms(rng: random.Random, n: int) -> list:
    alarms = []
    used = set()
    while len(alarms) < n:
        sub, prefix, comps = rng.choice(SUBSYSTEMS)
        comp = rng.choice(comps)
        mode = rng.choice(FAILURE_MODES)
        num = rng.randint(1, 999)
        code = f"{prefix}-{num:03d}" if prefix not in ("PLC", "SCD") else f"{prefix}-{rng.randint(1000, 9999)}"
        if code in used:
            continue
        used.add(code)
        title = f"{comp.capitalize()} {mode[0]}"
        alarms.append({
            "code": code, "subsystem": sub, "component": comp, "mode": mode[0], "title": title,
            "description": f"{mode[1]} on the {comp}.",
            "causes": mode[2], "actions": mode[3],
            "severity": rng.choice(["Warning", "Stop", "Emergency stop"]),
            "robots": rng.sample(ROBOTS, 3),
        })
    # cross-link related alarms (alarm relationship / root cause chains)
    for a in alarms:
        a["related"] = [b["code"] for b in rng.sample(alarms, 2) if b["code"] != a["code"]]
    return alarms


def write_alarm_manuals(out: str, alarms: list) -> None:
    by_sub = {}
    for a in alarms:
        by_sub.setdefault(a["subsystem"], []).append(a)
    for sub, items in by_sub.items():
        lines = [f"# {sub.upper()} Alarm Reference Manual", "", f"This chapter lists {sub} alarms for the paint shop.", ""]
        for i, a in enumerate(items, 1):
            lines += [
                f"## {i}. Alarm {a['code']} - {a['title']}",
                "",
                f"Severity: {a['severity']}. Typically reported on robots {', '.join(a['robots'])}.",
                f"Description: {a['description']}",
                "",
                "Possible causes:",
                *[f"- {c}" for c in a["causes"]],
                "",
                "Corrective action procedure:",
                *[f"{k}. {s}" for k, s in enumerate(a["actions"], 1)],
                "",
                f"Related alarms: {', '.join(a['related'])}. If {a['code']} occurs together with {a['related'][0]}, treat {a['related'][0]} as the probable root cause.",
                "",
            ]
        with open(os.path.join(out, f"alarm_manual_{sub}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def build_tags(rng: random.Random, n: int, alarms: list) -> list:
    measures = [("TEMP", "temperature", "C", (15, 260)), ("PRESS", "pressure", "bar", (0, 10)), ("FLOW", "flow rate", "cc/min", (0, 600)),
                ("SPEED", "speed", "rpm", (0, 60000)), ("HUM", "relative humidity", "%RH", (20, 90)), ("CURR", "motor current", "A", (0, 80)),
                ("VIB", "vibration velocity", "mm/s", (0, 25)), ("KV", "high voltage", "kV", (0, 100))]
    areas = [("OVN", "Oven"), ("BTH", "Booth"), ("PTR", "Pretreatment"), ("ED", "E-coat"), ("PK", "Paint kitchen"), ("RB", "Robot")]
    tags, used = [], set()
    while len(tags) < n:
        area, area_name = rng.choice(areas)
        unit_no = rng.randint(1, 6)
        zone = rng.randint(1, 8)
        m, mname, unit, (lo, hi) = rng.choice(measures)
        suffix = rng.choice(["PV", "SP", "AL"])
        name = f"{area}{unit_no}_Z{zone}_{m}_{suffix}"
        if name in used:
            continue
        used.add(name)
        hihi = round(lo + (hi - lo) * rng.uniform(0.8, 0.95), 1)
        high = round(hihi * 0.9, 1)
        tags.append({
            "tag": name,
            "description": f"{area_name} {unit_no} zone {zone} {mname} {'process value' if suffix == 'PV' else 'setpoint' if suffix == 'SP' else 'alarm status'}",
            "unit": unit, "plc_address": f"DB{rng.randint(100, 399)}.DBD{rng.randint(0, 500) * 4}",
            "plc": f"PLC-{rng.randint(1, 8):02d}", "high": high, "hihi": hihi,
            "alarm": rng.choice(alarms)["code"],
        })
    return tags


def write_tags(out: str, tags: list) -> None:
    with open(os.path.join(out, "tag_list.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tag", "Description", "Unit", "PLC", "PLC Address", "High Limit", "HiHi Limit", "Linked Alarm"])
        for t in tags:
            w.writerow([t["tag"], t["description"], t["unit"], t["plc"], t["plc_address"], t["high"], t["hihi"], t["alarm"]])


def write_defects(out: str) -> None:
    lines = ["# Paint Process Defect Troubleshooting Guide", ""]
    for i, (name, symptom, causes, actions, params) in enumerate(DEFECTS, 1):
        lines += [f"## {i}. Defect: {name}", "", f"Symptom: {symptom}.", "", "Root causes:", *[f"- {c}" for c in causes], "",
                  "Corrective actions:", *[f"{k}. {a}" for k, a in enumerate(actions, 1)], "", f"Key parameters to verify: {', '.join(params)}.", ""]
    with open(os.path.join(out, "process_defects.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_pm(out: str) -> None:
    lines = ["# Preventive Maintenance Procedures", ""]
    for i, (eq, tasks) in enumerate(PM_EQUIPMENT, 1):
        lines += [f"## {i}. PM-{i:03d} {eq} monthly maintenance", "", f"Interval: monthly. Lock out and tag out the {eq.lower()} before starting.", "",
                  *[f"{k}. {t}" for k, t in enumerate(tasks, 1)], "", "Record all readings in the CMMS work order.", ""]
    with open(os.path.join(out, "pm_procedures.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def iter_work_orders(rng: random.Random, n: int, alarms: list):
    for i in range(n):
        a = alarms[rng.randrange(len(alarms))]
        robot = rng.choice(a["robots"])
        cause = rng.choice(a["causes"])
        action = rng.choice(a["actions"])
        yield {
            "wo": f"WO-{2020 + i % 6}-{i:07d}",
            "date": f"{2020 + i % 6}-{rng.choice(MONTHS)}-{rng.randint(1, 28):02d}",
            "equipment": f"{robot} {a['component']}" if a["subsystem"] in ("servo", "atomizer", "motion") else a["component"],
            "zone": rng.choice(ZONES),
            "alarm": a["code"],
            "symptom": f"{a['title']} reported ({a['code']})",
            "root_cause": cause,
            "action": action,
            "downtime_min": rng.randint(5, 240),
        }


def write_work_orders(out: str, rng: random.Random, n: int, alarms: list) -> list:
    sample = []
    with open(os.path.join(out, "work_orders.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Work Order", "Date", "Equipment", "Zone", "Alarm", "Symptom", "Root Cause", "Action Taken", "Downtime (min)"])
        for wo in iter_work_orders(rng, n, alarms):
            w.writerow([wo["wo"], wo["date"], wo["equipment"], wo["zone"], wo["alarm"], wo["symptom"], wo["root_cause"], wo["action"], wo["downtime_min"]])
            if len(sample) < 400 and rng.random() < 0.05:
                sample.append(wo)
    return sample


NEGATIVES = [
    "How do I calibrate the spot welding gun tip dresser?",
    "What is the torque specification for the stamping press die clamp?",
    "Explain the procedure to replace the hydraulic accumulator on the injection moulding machine",
    "What is the recommended tyre pressure for the forklift?",
    "How do I configure the payroll export in SAP?",
    "What is the boiling point of the coolant used in the CNC lathe?",
    "Steps to reset the elevator emergency brake",
    "Which laser cutter nozzle should be used for 3 mm aluminium?",
]


def build_queries(rng: random.Random, alarms, tags, wo_sample) -> list:
    q = []
    for a in rng.sample(alarms, min(60, len(alarms))):
        q.append({"category": "alarm_code", "query": f"What does alarm {a['code']} mean and how do I fix it?", "expect": a["code"], "answer": a["actions"][0]})
    for a in rng.sample(alarms, min(40, len(alarms))):
        q.append({"category": "alarm_context", "query": f"Robot {a['robots'][0]} stopped with {a['code']}, what should I check first?", "expect": a["code"], "answer": a["actions"][0]})
    for a in rng.sample(alarms, min(40, len(alarms))):
        q.append({"category": "root_cause", "query": f"{a['code']} and {a['related'][0]} are active together, which one is the root cause?", "expect": a["code"], "answer": f"treat {a['related'][0]} as the probable root cause"})
    for t in rng.sample(tags, min(50, len(tags))):
        q.append({"category": "tag_limit", "query": f"What is the HiHi limit of {t['tag']}?", "expect": t["tag"], "answer": str(t["hihi"])})
    for t in rng.sample(tags, min(30, len(tags))):
        q.append({"category": "plc_address", "query": f"Which tag is mapped to {t['plc_address']} on {t['plc']}?", "expect": t["plc_address"], "answer": t["tag"]})
    for name, para in DEFECT_PARAPHRASES.items():
        q.append({"category": "process_semantic", "query": f"Painting problem: {para}. What are the causes?", "expect": f"Defect: {name}", "answer": dict((d[0], d[2][0]) for d in DEFECTS)[name]})
    for eq, _ in PM_EQUIPMENT:
        q.append({"category": "pm_procedure", "query": f"What is the monthly preventive maintenance for the {eq.lower()}?", "expect": f"{eq} monthly maintenance", "answer": dict(PM_EQUIPMENT)[eq][0]})
    for wo in wo_sample[:40]:
        q.append({"category": "history_wo", "query": f"What was the root cause and action for {wo['wo']}?", "expect": wo["wo"], "answer": wo["root_cause"]})
    for neg in NEGATIVES:
        q.append({"category": "negative", "query": neg, "expect": None})
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/data")
    ap.add_argument("--alarms", type=int, default=600)
    ap.add_argument("--tags", type=int, default=1500)
    ap.add_argument("--work-orders", type=int, default=8000)
    ap.add_argument("--no-queries", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    os.makedirs(args.out, exist_ok=True)
    alarms = build_alarms(rng, args.alarms)
    write_alarm_manuals(args.out, alarms)
    tags = build_tags(rng, args.tags, alarms)
    write_tags(args.out, tags)
    write_defects(args.out)
    write_pm(args.out)
    wo_sample = write_work_orders(args.out, rng, args.work_orders, alarms)
    if not args.no_queries:
        with open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8") as f:
            for item in build_queries(rng, alarms, tags, wo_sample):
                f.write(json.dumps(item) + "\n")
    print(f"corpus written to {args.out}: {len(alarms)} alarms, {len(tags)} tags, {args.work_orders} work orders")


if __name__ == "__main__":
    main()
