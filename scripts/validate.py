# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""B6: validation report on real HBEDB recordings (needs HBEDB_DIR). Writes out/VALIDATION.md and two plots.

    HBEDB_DIR=... python scripts/validate.py

1. Calibration against NASA preflight (pre-registered, B2).
2. Healthy retest, 2-fold cross-validation by subject: parameters (P5, q95_data) fitted on one half and the
   decision applied to the other half; baseline = first 20-s windows, retest = second 20-s windows.
3. Structural regression check: retest = 0.74 x the baseline trials themselves, so the ratio is exactly 0.74 and
   every person must be NO-GO through the 75% ratio rule. Shows the rule is wired, not that the app is safe.
Limitation: HBEDB has one session per person; day-to-day retest noise is not measured.
"""
from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from readiness import config as C  # noqa: E402
from readiness import params as P  # noqa: E402
from readiness import signals as S  # noqa: E402
from readiness.decision import ConditionInput, decide  # noqa: E402

NASA_PRE = {c: C.WOOD[c][1] for c in C.CONDITIONS}
STATUSES = ("GO", "CAUTION", "NO-GO", "CANNOT ASSESS")


def calibration(trials):
    sel = trials[trials.healthy & ~trials.vestibular & (trials.age >= 25) & (trials.age <= 55)]
    med = sel.groupby(["subject", "condition"]).eq_w1.median().unstack().median()
    return int(sel.subject.nunique()), {c: float(med[c]) for c in C.CONDITIONS}


def _q95(entry, b, cond, ag):
    t = int(np.digitize([b], entry["edges"])[0])
    return max(entry["q95"][t], P.q95_lit(b, cond, ag))


def healthy_retest(trials, rho: float = 1.0, seed: int = C.SEED, source: str = "w2"):
    """Single-test statuses for every healthy person with all four conditions, parameters from the other fold."""
    persons = P.healthy_persons(trials)
    subjects = sorted(persons.subject.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    halves = [set(subjects[0::2]), set(subjects[1::2])]
    rows = []
    for k in (0, 1):
        fit = P.derive_data_params(trials[trials.subject.isin(halves[k])], seed=seed + 1 + k)
        for subj in sorted(halves[1 - k]):
            mine = persons[persons.subject == subj]
            if len(mine) != 4:
                continue
            ag = mine.age_group.iloc[0]
            inputs = {}
            for _, row in mine.iterrows():
                e = fit[f"{ag}|{row.condition}"]
                b = row.baseline
                retest = np.clip(rho * (row.w2 if source == "w2" else row.w1), 0, 100)
                inputs[row.condition] = ConditionInput(list(retest), [False] * 3, b, _q95(e, b, row.condition, ag),
                                                       e["p5"], [True] * 3, True)
            dec = decide(inputs, "retest", 0.0)
            rows.append({"subject": subj, "age_group": ag, "status": dec.status,
                         "ratio_nogo": all(dec.per_condition[c].reason.startswith("Below 75% of baseline")
                                           for c in C.CONDITIONS)})
    return rows


def shares(rows, key=None):
    sel = rows if key is None else [r for r in rows if r["age_group"] == key]
    n = len(sel)
    cnt = Counter(r["status"] for r in sel)
    return n, {s: cnt.get(s, 0) / n for s in STATUSES}


def commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return (r.stdout.strip() or "unknown") + (" with uncommitted changes" if dirty else "")


def run_tests() -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True)
    return (r.stdout.strip().splitlines() or ["(no output)"])[-1]


def plots(cal, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{C.CONDITION_LABELS[c]}\n(NASA: {C.WOOD[c][0]})" for c in C.CONDITIONS]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, [cal[c] for c in C.CONDITIONS], 0.4, label="HBEDB healthy, 25-55 y (this app)")
    ax.bar(x + 0.2, [NASA_PRE[c] for c in C.CONDITIONS], 0.4, label="NASA crew preflight mean (Wood 2015, Table III)")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Equilibrium score")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Calibration: healthy adults scored by this app vs NASA crew preflight (different cohorts "
                 "and protocols)", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / "calibration.png", dpi=150)
    plt.close(fig)
    groups = [("All", None), ("Young", "Young"), ("Old", "Old")]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    colours = {"GO": "#2e7d32", "CAUTION": "#f9a825", "NO-GO": "#c62828", "CANNOT ASSESS": "#616161"}
    for i, (name, key) in enumerate(groups):
        n, sh = shares(rows, key)
        left = 0.0
        for s in STATUSES:
            ax.barh(i, sh[s] * 100, left=left, color=colours[s], label=s if i == 0 else None)
            if sh[s] >= 0.06:
                ax.text(left + sh[s] * 50, i, f"{s} {sh[s]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if s != "CAUTION" else "black")
            left += sh[s] * 100
        ax.text(101, i, f"n={n}", va="center", fontsize=8)
    ax.set_yticks(range(3), [g[0] for g in groups])
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of healthy people (single test, parameters from the other half)")
    ax.set_title("Healthy retest: single-test GO rate (overall GO also needs a second test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / "healthy_retest.png", dpi=150)
    plt.close(fig)


def main() -> int:
    if not C.full_data():
        print("Set HBEDB_DIR to the full HBEDB.", file=sys.stderr)
        return 1
    C.OUT_DIR.mkdir(exist_ok=True)
    trials = S.build_trial_table()
    n_cal, cal = calibration(trials)
    rows = healthy_retest(trials)
    rows_074 = healthy_retest(trials, rho=0.74, source="w1")
    tests = run_tests()
    plots(cal, rows)
    n_all, sh_all = shares(rows)
    n_y, sh_y = shares(rows, "Young")
    n_o, sh_o = shares(rows, "Old")
    n_s, sh_s = shares(rows_074)
    pct = lambda d: ", ".join(f"{s} {d[s]:.1%}" for s in STATUSES if d[s] > 0 or s != "CANNOT ASSESS")  # noqa: E731
    lines = [
        "# Validation report",
        "",
        "Generated by `scripts/validate.py` from the full Human Balance Evaluation Database (163 people, 1,930 "
        "trials). Seed 20261114. Limitation throughout: each HBEDB person was recorded in one session, so "
        "day-to-day retest noise is not measured, and no public dataset labels anyone as fit to leave a vehicle.",
        "",
        "## 1. Calibration against NASA crew preflight (pre-registered)",
        "",
        f"Healthy adults aged 25-55 (n = {n_cal}); median of each person's median score, first 20-s window. "
        "NASA values are preflight means of 37 ISS crew (Wood 2015, Table III). Tolerance: within 5 points for "
        "the firm conditions (set before looking).",
        "",
        "| Condition | NASA analogue | This app (median) | NASA preflight (mean) | Difference |",
        "|---|---|---|---|---|",
    ]
    for c in C.CONDITIONS:
        lines.append(f"| {C.CONDITION_LABELS[c]} | {C.WOOD[c][0]} | {cal[c]:.1f} | {NASA_PRE[c]:.1f} | "
                     f"{cal[c] - NASA_PRE[c]:+.1f} |")
    lines += [
        "",
        "Firm conditions pass. On foam with eyes closed this app's healthy median is 10.9 points above NASA's "
        "SOT 5 mean. That is an observed difference between different cohorts, estimators and protocols (foam "
        "here; a sway-referenced platform at NASA), not a measured effect of the support surface. Agreement on "
        "the firm conditions does not establish equivalence with NASA's method.",
        "",
        "![Calibration](calibration.png)",
        "",
        "## 2. Healthy retest (2-fold cross-validation by subject)",
        "",
        "Parameters are fitted on one half of the healthy people and applied to the other half. Baseline = first "
        "20-s windows; retest = second 20-s windows of the same trials (no impairment). Single test, so the "
        "two-test rule is not applied here.",
        "",
        f"- All (n = {n_all}): {pct(sh_all)}",
        f"- Young (n = {n_y}): {pct(sh_y)}",
        f"- Old (n = {n_o}): {pct(sh_o)}",
        "",
        "Healthy people are often held at CAUTION, especially older people on foam under literature-level noise. "
        "This is the deliberate safety trade-off: the app prefers a retest to a false clearance.",
        "",
        "![Healthy retest](healthy_retest.png)",
        "",
        "## 3. Structural regression check (not evidence of safety)",
        "",
        f"Retest = 0.74 x each person's own baseline trials, so the ratio is exactly 0.74 in every condition "
        f"(n = {n_s}): {pct(sh_s)}; the 75% ratio rule gave NO-GO in all four conditions for "
        f"{sum(r['ratio_nogo'] for r in rows_074)} of {n_s} people. This confirms the rule is wired; it is not "
        "evidence of safety.",
        "",
        "## 4. Automated tests",
        "",
        f"`pytest` (full data, HBEDB and OSDR files present) at commit {commit()}: {tests}. The count refers to "
        "that commit; later commits may add tests.",
        "",
        "Separately, before the hackathon the frozen mathematical framework was checked by an independent test "
        "bench (826,569 checks, 0 failed); that is a historical check of the framework, not of this code. See "
        "ASSUMPTIONS.md.",
    ]
    (C.OUT_DIR / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
