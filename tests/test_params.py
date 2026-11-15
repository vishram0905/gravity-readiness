# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Gate B3: committed parameters against MATH_FRAMEWORK §3 (no data needed) and re-derivation (full data)."""
import math

import numpy as np
import pytest

from readiness import config as C
from readiness import params as P

# MATH_FRAMEWORK §3 table: P5 and q95_data by baseline tertile
TABLE = {
    ("Young", "firm_eo"): (90.8, (1.051, 1.031, 1.029)),
    ("Young", "firm_ec"): (88.6, (1.059, 1.035, 1.022)),
    ("Young", "foam_eo"): (78.9, (1.099, 1.055, 1.053)),
    ("Young", "foam_ec"): (77.1, (1.086, 1.070, 1.039)),
    ("Old", "firm_eo"): (88.6, (1.047, 1.036, 1.028)),
    ("Old", "firm_ec"): (88.7, (1.047, 1.029, 1.024)),
    ("Old", "foam_eo"): (72.6, (1.104, 1.087, 1.066)),
    ("Old", "foam_ec"): (70.3, (1.113, 1.101, 1.067)),
}


@pytest.mark.parametrize("key", list(TABLE))
def test_table_matches_framework(key):
    ag, cond = key
    p5, q = TABLE[key]
    e = P.load_params()[f"{ag}|{cond}"]
    assert e["p5"] == pytest.approx(p5, abs=0.5)
    assert np.allclose(e["q95"], q, atol=0.01)


def test_drug_slope():
    assert P.drug_slope("foam_eo") == pytest.approx(1.87, abs=0.01)
    for c in ("firm_eo", "firm_ec", "foam_ec"):
        assert P.drug_slope(c) is None
    ci = P.load_params()["slope|foam_eo"]["ci"]
    assert ci[1] < 0


def test_lit_examples():
    assert P.q95_lit(85, "foam_ec", "Young") == pytest.approx(1.27, abs=0.01)
    assert P.q95_lit(85, "foam_ec", "Old") == pytest.approx(1.38, abs=0.01)


def test_lit_sd_conversion():
    assert P.lit_trial_sd("foam", "Young") == pytest.approx(13.1, abs=0.05)
    assert P.lit_trial_sd("foam", "Old") == pytest.approx(18.3, abs=0.05)
    assert P.lit_trial_sd("firm", "Young") == pytest.approx(2.2, abs=0.05)
    assert P.lit_trial_sd("firm", "Old") == pytest.approx(2.9, abs=0.05)


@pytest.mark.parametrize("ag", C.AGE_GROUPS)
@pytest.mark.parametrize("surf", ["firm", "foam"])
def test_lit_at_least_one_and_non_increasing(ag, surf):
    v = np.array(P.load_q95_lit()[f"{ag}|{surf}"])
    assert np.all(v >= 1.0)
    assert np.all(np.diff(v) <= 1e-12), f"max increase {np.diff(v).max():.2e}"


def test_noise_is_max_of_both():
    for (ag, cond) in TABLE:
        for b in (40.0, 70.0, 85.0, 93.0, 99.0):
            q = P.noise_q95(b, cond, ag)
            assert q == max(P.q95_data(b, cond, ag), P.q95_lit(b, cond, ag))
            assert q >= 1.0
    assert math.isnan(P.noise_q95(float("nan"), "foam_ec", "Young"))


@pytest.mark.full
def test_rederivation_matches_committed():
    from readiness import signals as S
    fresh = P.derive_data_params(S.build_trial_table())
    committed = P.load_params()
    for k, v in committed.items():
        if k.startswith("slope|"):
            assert fresh[k]["slope_per_decade"] == pytest.approx(v["slope_per_decade"], abs=1e-9)
        else:
            assert fresh[k]["p5"] == pytest.approx(v["p5"], abs=1e-9)
            assert np.allclose(fresh[k]["q95"], v["q95"], atol=1e-12)
