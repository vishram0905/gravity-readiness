# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""B7: choose the demo people by fixed rules (needs HBEDB_DIR) and copy only their files into data/demo/.

Rules (deterministic; no tuning of the maths):
- Candidates: all 12 trials present and valid.
- Young healthy: Young, healthy; the young person whose four baselines are closest to the young healthy medians
  among those for whom every demo scenario can be computed (a first GO exists within 10 days).
- Older healthy: Old, healthy; closest to the older healthy medians.
- Below the population floor: a healthy person whose baseline is at least 1 EQ point below P5 in at least one
  condition and whose healthy check is held at CAUTION by the floor; fewest such conditions, then closest to the
  medians (a clear but mild case, not a knife-edge one).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from readiness import config as C  # noqa: E402
from readiness import hbedb  # noqa: E402
from readiness import params as P  # noqa: E402
from readiness import scenarios as SC  # noqa: E402
from readiness import signals as S  # noqa: E402

KEEP_COLS = ["Trial", "Subject", "Vision", "Surface", "Age", "AgeGroup", "Gender", "Height", "Weight",
             "Illness2", "Disability"]


def main() -> int:
    if not C.full_data():
        print("Set HBEDB_DIR to the full HBEDB.", file=sys.stderr)
        return 1
    trials = S.build_trial_table()
    info = hbedb.load_info()
    ok = trials.groupby("subject").agg(n=("valid", "size"), v=("valid", "all"), w2=("eq_w2", lambda s: s.notna().all()))
    complete = set(ok[(ok.n == 12) & ok.v & ok.w2].index)
    base = trials.groupby(["subject", "condition"]).eq_w1.median().unstack()
    meta = trials.groupby("subject").agg(ag=("age_group", "first"), healthy=("healthy", "first"),
                                         vest=("vestibular", "first"))
    healthy = meta[meta.healthy & ~meta.vest].index

    def distance(subj, ag):
        ref = base.loc[[s for s in healthy if meta.loc[s, "ag"] == ag]].median()
        return float(np.sqrt(((base.loc[subj] - ref) ** 2).sum()))

    def ranked(ag, extra=lambda s: True):
        c = [s for s in complete if s in healthy and meta.loc[s, "ag"] == ag and extra(s)]
        return sorted(c, key=lambda s: (distance(s, ag), s))

    chosen = {}
    for subj in ranked("Young"):
        person = SC.load_person(int(subj))
        if SC.first_go_time(person) is not None:
            chosen["Young healthy"] = int(subj)
            break
    chosen["Older healthy"] = int(ranked("Old")[0])

    def n_below(s):
        ag = meta.loc[s, "ag"]
        return sum(base.loc[s, c] < P.p5(c, ag) - 1.0 for c in C.CONDITIONS)

    def floor_holds(s):
        res = SC.run_scenario(SC.load_person(int(s)), "Healthy check")
        return res.test.status == "CAUTION" and "population floor" in res.test.reason

    below = [s for s in complete if s in healthy and n_below(s) > 0 and floor_holds(s)]
    below.sort(key=lambda s: (n_below(s), distance(s, meta.loc[s, "ag"]), s))
    chosen["Below the population floor"] = int(below[0])

    demo = C.DEMO_DIR
    if demo.exists():
        shutil.rmtree(demo)
    demo.mkdir(parents=True)
    src = C.data_dir()
    rows = info[info.Subject.isin(chosen.values())]
    rows[KEEP_COLS].to_csv(demo / "BDSinfo.txt", sep="\t", index=False)
    for t in rows.Trial:
        for ext in (".hea", ".dat"):
            shutil.copy2(src / f"{t}{ext}", demo / f"{t}{ext}")
    (demo / "demo_people.json").write_text(json.dumps(chosen, indent=1))
    lines = ["# Demo people", "",
             "Three people from the Human Balance Evaluation Database (Santos and Duarte 2016, PhysioNet, ODC-By 1.0), "
             "chosen by fixed rules in `scripts/make_demo.py`. Only their trial files and a reduced info table are "
             "included.", "", "| Role | HBEDB subject | Age group | Baselines (firm EO, firm EC, foam EO, foam EC) |",
             "|---|---|---|---|"]
    for role, s in chosen.items():
        b = ", ".join(f"{base.loc[s, c]:.1f}" for c in C.CONDITIONS)
        lines.append(f"| {role} | {s} | {meta.loc[s, 'ag']} | {b} |")
    (demo / "DEMO_PEOPLE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(chosen), f"{len(rows)} trials copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
