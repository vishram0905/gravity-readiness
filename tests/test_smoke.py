# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
import importlib

MODULES = ["readiness", "readiness.config", "readiness.hbedb", "readiness.signals", "readiness.params",
           "readiness.decision", "readiness.forecast", "readiness.medicine", "readiness.scenarios"]


def test_every_module_imports():
    for name in MODULES:
        importlib.import_module(name)


def test_constants_match_framework():
    from readiness import config as C
    assert C.SEG_N == 2000 and C.MIN_COVERAGE == 0.99 and C.MIN_SWAY_SD_CM == 0.01
    assert C.COM_HEIGHT_FRAC == 0.55 and C.LOS_DEG == 12.5
    assert C.HEIGHT_CM == (100.0, 250.0) and C.MASS_KG == (30.0, 200.0)
    assert C.R_NOGO == 0.75 and C.MIN_TEST_GAP_H == 2.0 and C.C_REF == 11.25
    assert abs(C.AGE_EQUIV - 31 / 32) < 1e-12 and C.SEED == 20261114


def test_wood_csv_matches_config():
    import csv
    from readiness import config as C
    with open(C.PARAMS_DIR / "wood2015_table3.csv") as fh:
        rows = {r["condition"]: r for r in csv.DictReader(fh)}
    for cond, (sot, pre, r0, r0cl, tau, taucl) in C.WOOD.items():
        r = rows[cond]
        assert r["sot"] == sot
        assert [float(r[k]) for k in ("preflight", "r0", "r0_cl95", "tau_h", "tau_cl95")] == [pre, r0, r0cl, tau, taucl]
