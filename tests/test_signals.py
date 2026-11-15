# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Gate B2: COM and EQ unit tests (no data needed) and the pre-registered calibration (full data)."""
import math

import numpy as np
import pytest

from readiness import config as C
from readiness import signals as S

H = 170.0


def _sway(n=C.SEG_N, amp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / C.FS_HZ
    return amp * np.sin(2 * np.pi * 0.3 * t) + 0.01 * rng.standard_normal(n)


def test_eq_known_value():
    x = _sway()
    ptp = x.max() - x.min()
    expected = 100 * (1 - math.degrees(math.atan(ptp / (0.55 * H))) / 12.5)
    assert S.eq_window(x, H) == pytest.approx(expected, abs=1e-12)


def test_eq_completed_scales_and_floor():
    x = _sway()
    assert S.eq_window(x, H, completed=0.5) == pytest.approx(0.5 * S.eq_window(x, H))
    big = np.linspace(-60, 60, C.SEG_N)  # theta > 12.5 deg -> floor 0
    assert S.eq_window(big, H) == 0.0


@pytest.mark.parametrize("case", ["two_samples", "short", "missing", "flat", "inf_height", "short_height",
                                  "completed_15", "completed_nan", "completed_neg"])
def test_eq_invalid_is_nan(case):
    x = _sway()
    h, comp = H, 1.0
    if case == "two_samples":
        x = x[:2]
    elif case == "short":
        x = x[:1990]  # 19.9 s
    elif case == "missing":
        x = x.copy(); x[:30] = np.nan  # 1.5% missing
    elif case == "flat":
        x = np.full(C.SEG_N, 1.234)
    elif case == "inf_height":
        h = float("inf")
    elif case == "short_height":
        h = 50.0
    elif case == "completed_15":
        comp = 1.5
    elif case == "completed_nan":
        comp = float("nan")
    elif case == "completed_neg":
        comp = -0.1
    assert math.isnan(S.eq_window(x, h, completed=comp))


def test_one_percent_missing_still_valid():
    x = _sway().copy(); x[:20] = np.nan  # exactly 1% missing -> 99% valid
    assert math.isfinite(S.eq_window(x, H))


@pytest.mark.parametrize("h,m,ok", [(170, 70, True), (170, 10, False), (170, float("inf"), False),
                                    (170, float("nan"), False), (50, 70, False), (260, 70, False),
                                    (100, 30, True), (250, 200, True), (None, 70, False)])
def test_valid_anthro(h, m, ok):
    assert S.valid_anthro(h, m) is ok


def test_acceleration_units_constant_force():
    fx = np.full(100, 7.0)  # after offset removal -> 0
    assert np.allclose(S.body_acceleration(fx, 70.0), 0.0)
    fx = np.concatenate([np.full(50, 3.5), np.full(50, -3.5)])  # mean 0
    a = S.body_acceleration(fx, 70.0)
    assert a[0] == pytest.approx(-3.5 / 70.0 * 100.0)  # -0.05 m/s^2 = -5 cm/s^2


def test_glp_recovers_known_com():
    # Inverted-pendulum-like synthetic: COM x(t) sinusoid; Fx = -m a (subject's force on plate);
    # COP = x - a h_eff/g (any COP that equals COM where Fx = 0 works for GLP)
    fs, m = C.FS_HZ, 70.0
    t = np.arange(6000) / fs
    w = 2 * np.pi * 0.25
    x = 0.8 * np.sin(w * t) + 0.3 * np.sin(2 * np.pi * 0.11 * t + 0.4)
    a = -0.8 * w * w * np.sin(w * t) - 0.3 * (2 * np.pi * 0.11) ** 2 * np.sin(2 * np.pi * 0.11 * t + 0.4)
    fx = -m * a / 100.0 + 2.0  # N, plus a constant offset the method must remove
    cop = x - a * 0.9 / 9.81
    # constant offset removal subtracts mean(fx) = 2 + mean(-m a/100); keep mean(a) ~ 0 over 60 s
    com = S.com_glp(cop, fx, m, fs)
    ok = np.isfinite(com)
    assert ok.mean() > 0.9
    assert np.max(np.abs(com[ok] - x[ok])) < 0.05  # cm


def test_glp_sign_matters():
    fs, m = C.FS_HZ, 70.0
    t = np.arange(6000) / fs
    w = 2 * np.pi * 0.25
    x = 0.8 * np.sin(w * t)
    a = -0.8 * w * w * np.sin(w * t)
    cop = x - a * 0.9 / 9.81
    good = S.com_glp(cop, -m * a / 100.0, m, fs)
    flipped = S.com_glp(cop, m * a / 100.0, m, fs)
    ok = np.isfinite(good) & np.isfinite(flipped)
    assert np.max(np.abs(good[ok] - x[ok])) < 0.02
    assert np.max(np.abs(flipped[ok] - x[ok])) > 0.2  # wrong sign is detectably wrong


def test_nan_outside_crossings():
    f = np.concatenate([np.full(10, 5.0), np.sin(np.linspace(0, 6 * np.pi, 200)) * 3, np.full(10, 5.0)])
    com = S.com_glp(np.zeros_like(f), f, 70.0)
    zc = S.zero_crossings(f - f.mean())
    assert np.all(np.isnan(com[:zc[0]])) and np.all(np.isnan(com[zc[-1] + 1:]))


@pytest.fixture(scope="module")
def fresh_table():
    # Always recompute through load_trial and trial_scores (never the cache), so regressions are caught
    return S.build_trial_table(force=True)


@pytest.mark.full
def test_calibration_preregistered(fresh_table):
    df = fresh_table
    sel = df[df.healthy & (df.age >= 25) & (df.age <= 55)]
    assert sel.subject.nunique() == 43
    per = sel.groupby(["subject", "condition"]).eq_w1.median().unstack()
    med = per.median()
    assert med["firm_eo"] == pytest.approx(94.1, abs=0.5)
    assert med["firm_ec"] == pytest.approx(92.9, abs=0.5)
    assert abs(med["firm_eo"] - 94.3) <= 5 and abs(med["firm_ec"] - 89.9) <= 5
    assert med["foam_eo"] == pytest.approx(86.9, abs=1.0)
    assert med["foam_ec"] == pytest.approx(85.8, abs=1.0)


@pytest.mark.full
def test_trial_table_complete(fresh_table):
    df = fresh_table
    assert len(df) == 1930
    assert df.valid.mean() > 0.99


@pytest.mark.full
def test_cache_equals_fresh(fresh_table):
    cached = S.build_trial_table(force=False)
    assert np.allclose(cached.eq_w1, fresh_table.eq_w1, equal_nan=True)
    assert np.allclose(cached.eq_w2, fresh_table.eq_w2, equal_nan=True)


def test_lpf_keyword_only():
    x = np.sin(2 * np.pi * 0.3 * np.arange(6000) / 100)
    with pytest.raises(TypeError):
        S.com_lpf(x, 100, 170)  # positional order cannot be confused
    y = S.com_lpf(x, fs=100, height_cm=170)
    assert y.shape == x.shape and np.max(np.abs(y)) < 1.0
