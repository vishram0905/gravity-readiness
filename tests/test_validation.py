# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Gate B6: real-data validation numbers within tolerance of the pre-event verification."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytestmark = pytest.mark.full


@pytest.fixture(scope="module")
def trials():
    from readiness import signals as S
    return S.build_trial_table(force=True)


def test_healthy_retest_rates(trials):
    import validate as V
    rows = V.healthy_retest(trials)
    n, sh = V.shares(rows)
    assert n == 140
    assert sh["GO"] == pytest.approx(0.429, abs=0.03)
    assert sh["CAUTION"] == pytest.approx(0.564, abs=0.03)
    assert sh["NO-GO"] == pytest.approx(0.007, abs=0.03)
    assert sh["CANNOT ASSESS"] == 0.0


def test_structural_check_no_go(trials):
    import validate as V
    rows = V.healthy_retest(trials, rho=0.74, source="w1")
    n, sh = V.shares(rows)
    assert n == 140 and sh["NO-GO"] == 1.0
    assert all(r["ratio_nogo"] for r in rows)  # through the 75% ratio rule, in every condition


def test_calibration_in_report(trials):
    import validate as V
    n, cal = V.calibration(trials)
    assert n == 43 and abs(cal["firm_eo"] - 94.3) <= 5 and abs(cal["firm_ec"] - 89.9) <= 5
