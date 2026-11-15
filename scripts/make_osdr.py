# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Processed subset of NASA OSDR Inspiration4 crew blood tests (OSD-569 CBC, OSD-575 CMP) -> data/osdr/.

Usage: python scripts/make_osdr.py [--src DIR]   (downloads the two OSDR files if --src is not given)
Keeps seven analytes, drops the sex column, merges the female-specific analyte names, and writes
data/osdr/inspiration4_labs.csv (one row per crew member, analyte and time point).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from readiness import crew_labs as L  # noqa: E402

BASE = "https://osdr.nasa.gov/geode-py/ws/studies/{osd}/download?source=datamanager&file={f}"
FILES = {  # study -> (file, time-point column)
    "OSD-569": ("LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv", "TEST_DATE"),
    "OSD-575": ("LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv", "timepoint"),
}


def _read(osd: str, src: Path | None) -> list[dict]:
    fname, _ = FILES[osd]
    if src is not None:
        text = (src / fname).read_text(encoding="utf-8-sig")
    else:
        with urllib.request.urlopen(BASE.format(osd=osd, f=fname), timeout=60) as r:
            text = r.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None)
    a = ap.parse_args()
    out = []
    for osd, (_, tcol) in FILES.items():
        for r in _read(osd, a.src):
            name = r["ANALYTE"].replace(" (FEMALE)", "")
            if name not in L.ANALYTES:
                continue
            out.append({"study": osd, "subject": r["SUBJECT_ID"], "analyte": name, "units": r["UNITS"],
                        "timepoint": r[tcol], "value": r["VALUE"], "range_min": r["RANGE_MIN"],
                        "range_max": r["RANGE_MAX"]})
    out.sort(key=lambda x: (x["analyte"], x["subject"], L.TIMEPOINTS.index(x["timepoint"])))
    L.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with L.DATA_FILE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {L.DATA_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
