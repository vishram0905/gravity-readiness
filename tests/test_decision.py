# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Gate B4: decision rules against an independent statement of the framework arithmetic."""
import math

import numpy as np
import pytest

from readiness import config as C
from readiness.decision import (CA, CAUTION, GO, NOGO, ConditionInput, Decision, decide, decide_condition,
                                decide_confirmed)
from readiness.decision import ConditionResult as ConditionResultForTest

RANK = C.STATUS_RANK


def reference(eqs, fell, b, q95, p5, valid, bvalid):
    """MATH_FRAMEWORK §4 written out directly (independent of the implementation)."""
    if len(eqs) < 3 or len(valid) != len(eqs) or len(fell) != len(eqs):
        return CA
    if not all(valid) or not bvalid:
        return CA
    if not (math.isfinite(b) and 0 < b <= 100) or not (math.isfinite(q95) and q95 >= 1) or not math.isfinite(p5):
        return CA
    if any((not math.isfinite(e)) or e < 0 or e > 100 for e in eqs):
        return CA
    s = sorted(eqs)[len(eqs) // 2] if len(eqs) % 2 else float(np.median(eqs))
    if any(fell):
        return NOGO
    r = s / b
    if r < 0.75:
        return NOGO
    if s < p5:
        return CAUTION
    return GO if r / q95 >= 0.75 else CAUTION


def ci(eqs=(90, 90, 90), fell=(False,) * 3, b=90.0, q95=1.05, p5=80.0, valid=(True,) * 3, bvalid=True):
    as_list = lambda x: None if x is None else list(x)  # noqa: E731
    return ConditionInput(as_list(eqs), as_list(fell), b, q95, p5, as_list(valid), bvalid)


def test_random_inputs_match_framework():
    rng = np.random.default_rng(C.SEED)
    n_checked = 0
    for _ in range(200_000):
        k = 3 if rng.random() < 0.9 else int(rng.integers(0, 6))
        eqs = list(rng.uniform(-5, 105, k)) if rng.random() < 0.05 else list(rng.uniform(0, 100, k))
        if rng.random() < 0.02 and k:
            eqs[0] = float("nan")
        fell = list(rng.random(k) < 0.05)
        valid = list(rng.random(k) > 0.03)
        b = float(rng.uniform(-10, 110)) if rng.random() < 0.05 else float(rng.uniform(20, 100))
        q95 = float(rng.uniform(0.9, 1.5))
        p5 = float(rng.uniform(50, 95)) if rng.random() > 0.01 else float("nan")
        bvalid = bool(rng.random() > 0.02)
        got = decide_condition(ConditionInput(eqs, fell, b, q95, p5, valid, bvalid)).status
        assert got == reference(eqs, fell, b, q95, p5, valid, bvalid), (eqs, fell, b, q95, p5, valid, bvalid)
        n_checked += 1
    assert n_checked == 200_000


def test_fall_with_good_median_is_nogo():
    assert decide_condition(ci(eqs=(0, 80, 80), fell=(True, False, False), b=82)).status == NOGO


def test_monotone_in_scores():
    rng = np.random.default_rng(1)
    for _ in range(20_000):
        b, q95, p5 = rng.uniform(40, 100), rng.uniform(1, 1.4), rng.uniform(40, 95)
        e = rng.uniform(0, 100, 3)
        j = int(rng.integers(3))
        bump = e.copy()
        bump[j] = min(100.0, bump[j] + rng.uniform(0, 20))
        s1 = decide_condition(ci(eqs=e, b=b, q95=q95, p5=p5)).status
        s2 = decide_condition(ci(eqs=bump, b=b, q95=q95, p5=p5)).status
        assert RANK[s2] <= RANK[s1]


def test_caution_region():
    b, q95, p5 = 80.0, 1.2, 70.0
    upper = max(0.75 * q95, p5 / b)  # 0.9
    for r in np.linspace(0.70, 1.10, 401):
        s = decide_condition(ci(eqs=(r * b,) * 3, b=b, q95=q95, p5=p5)).status
        if r < 0.75:
            assert s == NOGO
        elif r < upper - 1e-12:
            assert s == CAUTION
        elif r >= upper + 1e-12:
            assert s == GO
    # floor active: P5/b above 0.75 q95
    b, q95, p5 = 90.0, 1.02, 85.0
    assert decide_condition(ci(eqs=(84.9,) * 3, b=b, q95=q95, p5=p5)).status == CAUTION
    assert decide_condition(ci(eqs=(85.1,) * 3, b=b, q95=q95, p5=p5)).status == GO


@pytest.mark.parametrize("kw", [dict(valid=None), dict(fell=None), dict(valid=(True, True)), dict(eqs=(90, 90)),
                                dict(valid=(1, 1, 1)), dict(bvalid=None), dict(b=0.0), dict(b=101.0),
                                dict(b=float("nan")), dict(q95=0.99), dict(q95=float("nan")),
                                dict(p5=float("nan")), dict(eqs=(90, 90, 101)), dict(eqs=(90, 90, -1)),
                                dict(eqs=(90, 90, float("nan"))), dict(valid=(True, False, True)),
                                dict(bvalid=False)])
def test_invalid_is_cannot_assess(kw):
    assert decide_condition(ci(**kw)).status == CA


def test_validity_before_fall():
    assert decide_condition(ci(fell=(True, False, False), valid=(True, False, True))).status == CA


def _session(status_by_cond=None, sid="s1", t=10.0, drop=None):
    inputs = {c: ci() for c in C.CONDITIONS}
    if drop:
        inputs.pop(drop)
    return decide(inputs, sid, t)


def test_overall_worst_and_missing_caps():
    good = _session()
    assert good.status == GO and good.measured
    inputs = {c: ci() for c in C.CONDITIONS}
    inputs["foam_ec"] = ci(eqs=(0, 90, 90), fell=(True, False, False))
    assert decide(inputs, "s", 1.0).status == NOGO
    assert _session(drop="foam_ec").status == CAUTION
    inputs = {c: ci() for c in C.CONDITIONS}
    inputs["firm_eo"] = ci(valid=(True, True, False))
    assert decide(inputs, "s", 1.0).status == CA
    assert decide({}, "s", 1.0).status == CA


def test_two_test_rule():
    a = _session(sid="a", t=10.0)
    b = _session(sid="b", t=12.0)
    assert decide_confirmed([a]).status == CAUTION
    assert decide_confirmed([a, b]).status == GO
    assert decide_confirmed([a, a]).status == CA                                   # duplicate (tied time)
    assert decide_confirmed([a, _session(sid="a", t=12.0)]).status == CAUTION       # same session
    assert decide_confirmed([a, _session(sid="b", t=11.99)]).status == CAUTION      # < 2 h
    assert decide_confirmed([b, a]).status == CA                                   # out of order
    assert decide_confirmed([a, _session(sid="b", t=None)]).status == CA            # missing time
    assert decide_confirmed([_session(sid=None, t=10), _session(sid="b", t=12)]).status == CAUTION
    unmeasured = Decision(GO, "forecast", {}, False, "c", 14.0)
    assert decide_confirmed([a, b, unmeasured]).status == CA                       # unmeasured latest
    unmeasured_mid = Decision(GO, "forecast", {}, False, "c", 11.0)
    assert decide_confirmed([a, unmeasured_mid, b]).status == CAUTION              # unmeasured previous
    nogo = decide({c: ci(eqs=(0, 0, 0)) for c in C.CONDITIONS}, "n", 20.0)
    assert decide_confirmed([a, b, nogo]).status == NOGO                           # never better than latest
    assert decide_confirmed([]).status == CA


def _status_session(status, sid, t):
    eqs = {GO: (90, 90, 90), CAUTION: (70, 70, 70), NOGO: (0, 0, 0)}
    if status == CA:
        inputs = {c: ci(valid=(True, True, False)) for c in C.CONDITIONS}
    else:
        inputs = {c: ci(eqs=eqs[status], p5=80.0 if status == CAUTION else 60.0) for c in C.CONDITIONS}
    d = decide(inputs, sid, t)
    assert d.status == status
    return d


@pytest.mark.parametrize("prev", [CAUTION, NOGO, CA])
def test_previous_not_go_blocks_confirmation(prev):
    p = _status_session(prev, "p", 10.0)
    g = _status_session(GO, "g", 13.0)
    assert decide_confirmed([p, g]).status == CAUTION


def test_longer_histories_and_ties():
    nogo14 = _status_session(NOGO, "n", 14.0)
    go10, go12 = _status_session(GO, "a", 10.0), _status_session(GO, "b", 12.0)
    assert decide_confirmed([nogo14, go10, go12]).status == CA        # out of order anywhere
    assert decide_confirmed([go10, go12, nogo14]).status == NOGO      # chronological
    nogo10 = _status_session(NOGO, "n", 10.0)
    assert decide_confirmed([nogo10, go10, go12]).status == CA        # tied times
    assert decide_confirmed([go10, nogo10, go12]).status == CA
    early_nogo = _status_session(NOGO, "n", 8.0)
    assert decide_confirmed([early_nogo, go10, go12]).status == GO    # two most recent are GO


def test_exact_boundaries():
    # r = 0.75 exactly is not NO-GO (60/80 = 0.75 exactly in binary floating point)
    assert decide_condition(ci(eqs=(60, 60, 60), b=80.0, q95=1.2, p5=50.0)).status == CAUTION
    # r / q95 = 0.75 exactly is GO, with q95 = 1 allowed
    assert decide_condition(ci(eqs=(60, 60, 60), b=80.0, q95=1.0, p5=50.0)).status == GO
    assert decide_condition(ci(eqs=(75, 75, 75), b=80.0, q95=1.25, p5=50.0)).status == GO  # 0.9375/1.25
    # score = P5 exactly is not below the floor
    assert decide_condition(ci(eqs=(75, 75, 75), b=80.0, q95=1.0, p5=75.0)).status == GO
    assert decide_condition(ci(eqs=(74.99, 74.99, 74.99), b=80.0, q95=1.0, p5=75.0)).status == CAUTION


@pytest.mark.parametrize("bad", ["999", b"999", "90,90,90", [[90, 90, 90]], 90.0, [90, "x", 90]])
def test_malformed_scores_cannot_assess(bad):
    r = decide_condition(ConditionInput(bad, [False] * 3, 90.0, 1.05, 80.0, [True] * 3, True))
    assert r.status == CA


@pytest.mark.parametrize("field", ["valid", "fell"])
def test_malformed_flags_cannot_assess(field):
    kw = {"valid": [True] * 3, "fell": [False] * 3}
    kw[field] = "TTT"
    r = decide_condition(ConditionInput([90, 90, 90], kw["fell"], 90.0, 1.05, 80.0, kw["valid"], True))
    assert r.status == CA


def test_reasons_state_boundaries_plainly():
    at = decide_condition(ci(eqs=(67.5,) * 3, b=90.0, q95=1.05, p5=60.0))
    assert at.status == CAUTION and at.reason.startswith("At or above 75% of baseline, but")
    below = decide_condition(ci(eqs=(67.49999,) * 3, b=90.0, q95=1.05, p5=60.0))
    assert below.status == NOGO and "ratio 0.7499, below 0.75" in below.reason and "0.7500" not in below.reason


def test_iterators_rejected_and_scores_consistent():
    r = decide_condition(ConditionInput(iter([90.0, 90.0, 90.0]), [False] * 3, 90.0, 1.05, 80.0, [True] * 3, True))
    assert r.status == CA
    r = decide_condition(ConditionInput((x for x in [90.0] * 3), [False] * 3, 90.0, 1.05, 80.0, [True] * 3, True))
    assert r.status == CA
    r = decide_condition(ConditionInput([90.0] * 3, iter([False] * 3), 90.0, 1.05, 80.0, [True] * 3, True))
    assert r.status == CA
    ok = decide_condition(ConditionInput(np.array([90.0, 90.0, 90.0]), [False] * 3, 90.0, 1.05, 80.0, [True] * 3, True))
    assert ok.status == GO and ok.score == 90.0


@pytest.mark.parametrize("kind", ["fell_flat", "valid_flat", "fell_iter_list", "complex_scores", "complex_list",
                                  "bool_scores", "object_scores", "2d_scores", "int_flags", "complex_baseline",
                                  "complex_q95", "bool_baseline"])
def test_malformed_inputs_never_go(kind):
    eqs, fell, valid, b, q95, p5 = [90.0] * 3, [False] * 3, [True] * 3, 90.0, 1.05, 80.0
    if kind == "fell_flat":
        fell = np.array([False, True, False]).flat
    elif kind == "valid_flat":
        valid = np.array([True, False, True]).flat
    elif kind == "fell_iter_list":
        fell = iter([False, True, False])
    elif kind == "complex_scores":
        eqs = np.array([90 + 1j] * 3)
    elif kind == "complex_list":
        eqs = [90 + 1j] * 3
    elif kind == "bool_scores":
        eqs = [True, True, True]
    elif kind == "object_scores":
        eqs = np.array([90.0, 90.0, 90.0], dtype=object)
    elif kind == "2d_scores":
        eqs = np.array([[90.0, 90.0, 90.0]])
    elif kind == "int_flags":
        fell = [0, 1, 0]
    elif kind == "complex_baseline":
        b = np.complex128(90 + 0j)
    elif kind == "complex_q95":
        q95 = 1.05 + 0j
    elif kind == "bool_baseline":
        b = True
    r = decide_condition(ConditionInput(eqs, fell, b, q95, p5, valid, True))
    assert r.status == CA, (kind, r)


def test_array_inputs_still_accepted():
    r = decide_condition(ConditionInput(np.array([90, 90, 90]), np.array([False, False, False]), np.float64(90.0),
                                        np.float64(1.05), 80, np.array([True, True, True]), np.bool_(True)))
    assert r.status == GO
    r = decide_condition(ConditionInput((90.0, 90.0, 90.0), (True, False, False), 90, 1.05, 80.0, (True,) * 3, True))
    assert r.status == NOGO


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize("kind", ["masked_fell", "masked_scores", "masked_valid", "matrix_scores", "masked_element"])
def test_array_subclasses_rejected(kind):
    eqs, fell, valid = [90.0] * 3, [False] * 3, [True] * 3
    if kind == "masked_fell":
        fell = np.ma.array([False, True, False], mask=[False, True, False])
    elif kind == "masked_scores":
        eqs = np.ma.array([90.0, 90.0, 500.0], mask=[False, False, True])
    elif kind == "masked_valid":
        valid = np.ma.array([True, False, True], mask=[False, True, False])
    elif kind == "matrix_scores":
        eqs = np.matrix([90.0, 90.0, 90.0])
    elif kind == "masked_element":
        eqs = [90.0, 90.0, np.ma.masked]
    r = decide_condition(ConditionInput(eqs, fell, 90.0, 1.05, 80.0, valid, True))
    assert r.status == CA, (kind, r)
    s = decide({c: ConditionInput(eqs, fell, 90.0, 1.05, 80.0, valid, True) for c in C.CONDITIONS}, "a", 1.0)
    assert s.status == CA


def test_hand_made_go_decisions_not_trusted():
    fake1 = Decision(GO, "fake", {}, True, "a", 10.0)
    fake2 = Decision(GO, "fake", {}, True, "b", 12.0)
    assert decide_confirmed([fake1, fake2]).status == CA
    real1 = _session(sid="a", t=10.0)
    assert decide_confirmed([fake1, _session(sid="b", t=12.0)]).status == CAUTION   # previous not trusted
    assert decide_confirmed([real1, fake2]).status == CA                              # latest not trusted
    partial = Decision(GO, "partial", {"firm_eo": real1.per_condition["firm_eo"]}, True, "b", 12.0)
    assert decide_confirmed([real1, partial]).status == CA


def test_fabricated_complete_go_not_trusted():
    pc = {c: ConditionResultForTest(GO, "fake", 0, 0, 0) for c in C.CONDITIONS}
    a = Decision(GO, "fake", pc, True, "a", 0.0)
    b = Decision(GO, "fake", pc, True, "b", 2.0)
    assert decide_confirmed([a, b]).status == CA
    # genuine decision with its per-condition labels tampered: recomputation from inputs decides
    real_a, real_b = _session(sid="a", t=0.0), _session(sid="b", t=2.0)
    nogo_inputs = {c: ci(eqs=(0, 0, 0)) for c in C.CONDITIONS}
    tampered = Decision(GO, "tampered", real_b.per_condition, True, "b", 2.0, nogo_inputs)
    assert decide_confirmed([real_a, tampered]).status == CA
    assert decide_confirmed([real_a, real_b]).status == GO


@pytest.mark.parametrize("sid", [float("nan"), "", "  ", 3, None, b"a"])
def test_bad_session_ids(sid):
    a = _session(sid=sid, t=0.0)
    b = _session(sid=sid, t=2.0)
    assert decide_confirmed([a, b]).status == CAUTION
    assert decide_confirmed([_session(sid="x", t=0.0), _session(sid=sid, t=2.0)]).status == CAUTION


@pytest.mark.parametrize("flag", ["False", float("nan"), [False], 1, np.bool_(True)])
def test_measured_must_be_true(flag):
    from dataclasses import replace
    a, b = _session(sid="a", t=0.0), _session(sid="b", t=2.0)
    assert decide_confirmed([replace(a, measured=flag), b]).status == CAUTION
    assert decide_confirmed([a, replace(b, measured=flag)]).status == CA


def test_scalar_converted_once():
    class Shifty(float):
        n = 0

        def __float__(self):
            Shifty.n += 1
            return 2.0 if Shifty.n < 3 else 1.0

    r = decide_condition(ConditionInput([90.0] * 3, [False] * 3, 90.0, Shifty(2.0), 80.0, [True] * 3, True))
    assert r.status == CA  # subclass rejected outright
    for q in (np.float64(2.0), 2, 2.0):
        assert decide_condition(ConditionInput([90.0] * 3, [False] * 3, 90.0, q, 80.0, [True] * 3, True)).status == CAUTION
