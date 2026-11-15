# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Gate B5: retest forecast and promethazine scenario."""
import itertools
import math

import numpy as np
import pytest

from readiness import config as C
from readiness import forecast as F
from readiness import medicine as M


def test_typical_crew_foam_ec():
    lo, mid, hi = F.typical_crew("foam_ec", 0.9)
    assert mid == pytest.approx(40.9, abs=0.15)
    assert lo == pytest.approx(14.7, abs=0.15)
    assert hi == pytest.approx(72.3, abs=0.15)


def test_typical_crew_firm_eo_already_at_target():
    assert F.typical_crew("firm_eo", 0.9) == (0.0, 0.0, 0.0)
    for cond in C.CONDITIONS:
        for q in (0.5, 0.75, 0.9, 0.97, 0.999):
            assert all(v >= 0 for v in F.typical_crew(cond, q))


def test_unreachable_target():
    assert F.typical_crew("foam_ec", 1.0) is None
    assert F.retest_hours(0.5, 1.02, 19.4) is None
    assert M.t_go(0.5, 19.4, 1.0) is None


def test_forecast_reaches_q_exactly():
    for cond in C.CONDITIONS:
        w = F.wood(cond)
        for r_m, q in ((0.5, 0.8), (0.76, 0.9), (0.2, 0.97)):
            dt = F.retest_hours(r_m, q, w["tau"])
            D_i = 1 - r_m
            assert F.recovery(dt, D_i, w["tau"]) == pytest.approx(q, abs=1e-12)
    assert F.retest_hours(0.95, 0.9, 19.4) == 0.0


def test_retest_range_ordered():
    lo, mid, hi = F.retest_range(0.6, 0.9, "foam_ec")
    assert lo < mid < hi


def test_pk_anchors():
    for route in ("oral", "im"):
        t_half, tmax, cmax = C.GANDIA[route]
        t = np.linspace(0, 48, 480_001)
        c = M.conc(t, 50.0, route)
        assert t[np.argmax(c)] == pytest.approx(tmax, abs=0.1)
        assert c.max() == pytest.approx(cmax, rel=1e-6)
        ke, ka, _ = M.pk(route)
        assert ke == pytest.approx(math.log(2) / t_half)
        assert math.log(ka / ke) / (ka - ke) == pytest.approx(tmax, abs=1e-9)
    t = np.linspace(0, 48, 480_001)
    assert M.c_rel(t, 25.0, "oral").max() == pytest.approx(1.0, rel=1e-6)
    assert M.c_rel(t, 25.0, "im").max() == pytest.approx(2.33, abs=0.01)
    assert M.conc(-1.0, 25.0, "oral") == 0.0


def test_drug_never_raises_r():
    t = np.linspace(-5, 240, 5000)
    for dose, route, b_c in itertools.product(C.DOSES_MG, C.ROUTES, (None, 1.87, 15.0)):
        d = M.delta(t, dose, route, b_c, 80.0, t_dose=2.0)
        assert np.all(d >= 0)
        if b_c is None or dose == 0:
            assert np.all(d == 0)


def test_initial_deficit_clamped():
    assert M.initial_deficit(1.05, 10.0, 19.4) == 0.0
    assert M.initial_deficit(0.5, 0.0, 19.4) == pytest.approx(0.5)


def _brute(D, tau, q, dose, route, b_c, b, t_dose, t_start, step=0.002):
    t_end = t_start + 12 * tau + 60
    t = np.arange(t_start, t_end + step, step)
    r = M.curve(t, D, tau, dose, route, b_c, b, t_dose)
    below = np.flatnonzero(r < q)
    if below.size == 0:
        return t_start
    i = below[-1]
    assert i < t.size - 1, "horizon too short"
    return t[i]


def test_t_go_matches_dense_scan():
    n = 0
    worst = 0.0
    grid = itertools.product((-0.3, 0.0, 0.05, 0.3, 0.824, 1.105), (8.7, 19.4, 44.6, 116.1), (0.8, 0.9, 0.97),
                             ((0.0, "oral"), (12.5, "oral"), (25.0, "oral"), (50.0, "oral"), (25.0, "im"),
                              (50.0, "im")), (None, 1.87, 15.0))
    for D, tau, q, (dose, route), b_c in grid:
        for t_dose, t_start in ((0.0, 0.0), (3.0, 1.0), (-2.0, 0.5)):
            got = M.t_go(D, tau, q, dose, route, b_c, 80.0, t_dose, t_start)
            ref = _brute(D, tau, q, dose, route, b_c, 80.0, t_dose, t_start)
            err = abs(got - ref)
            worst = max(worst, err)
            assert err <= 0.004, (D, tau, q, dose, route, b_c, t_dose, t_start, got, ref)
            n += 1
    assert n == 6 * 4 * 3 * 6 * 3 * 3


def test_t_go_without_drug_equals_closed_form():
    for D, tau, q in itertools.product((0.05, 0.3, 0.824), (19.4, 44.6), (0.8, 0.9)):
        exp = F._typical(D, tau, q)
        assert M.t_go(D, tau, q) == pytest.approx(exp, abs=1e-12)


def test_dose_moves_estimate_later():
    # foam-EO is the only condition with an established age slope
    D, tau, q = 0.4, F.wood("foam_eo")["tau"], 0.9
    base = M.t_go(D, tau, q)
    with_drug = M.t_go(D, tau, q, 25.0, "oral", 1.87, 80.0, t_dose=0.0)
    assert with_drug > base


def test_regression_extremum_on_grid_point():
    # independent check case: derivative zero exactly on a grid point; correct answer 12.3765 h, not 0 h
    got = M.t_go(0.029400653805575126, 44.6, 0.97, 12.5, "oral", 1.87, 80.0, 0.0, 0.0)
    assert got == pytest.approx(12.3765074664, abs=1e-6)


def test_split_time_uses_tmax_not_cmax():
    assert M.split_time("oral", 1.0) == pytest.approx(1.0 + 2.9)
    assert M.split_time("im", 1.0) == pytest.approx(1.0 + 2.0)
    assert M.split_time("oral", -5.0, t_start=0.0) == 0.0
    # after dose + tmax the curve must be strictly increasing (the solver's tail argument)
    for route in C.ROUTES:
        tmax = C.GANDIA[route][1]
        t = np.linspace(1.0 + tmax, 1.0 + tmax + 200, 20001)
        r = M.curve(t, 0.3, 19.4, 50.0, route, 15.0, 80.0, t_dose=1.0)
        assert np.all(np.diff(r) > 0)


def test_t_go_random_scenarios_against_dense_scan():
    rng = np.random.default_rng(C.SEED)
    worst = 0.0
    for _ in range(1500):
        D = float(rng.uniform(-0.1, 1.2))
        tau = float(rng.uniform(5, 150))
        q = float(rng.uniform(0.6, 0.995))
        dose = float(rng.choice([12.5, 25.0, 50.0]))
        route = str(rng.choice(["oral", "im"]))
        b_c = float(rng.uniform(0.5, 20))
        b = float(rng.uniform(40, 100))
        t_dose = float(rng.uniform(-5, 30))
        t_start = float(rng.uniform(0, 30))
        got = M.t_go(D, tau, q, dose, route, b_c, b, t_dose, t_start)
        ref = _brute(D, tau, q, dose, route, b_c, b, t_dose, t_start)
        err = abs(got - ref)
        worst = max(worst, err)
        assert err <= 0.004, (D, tau, q, dose, route, b_c, b, t_dose, t_start, got, ref)
        # property: never below target after the returned time
        tt = np.linspace(got + 1e-6, got + 10 * tau + 50, 20001)
        assert np.all(M.curve(tt, D, tau, dose, route, b_c, b, t_dose) >= q - 1e-12)
