# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Gate B1: HBEDB reader against the full database."""
import numpy as np
import pytest

from readiness import config as C
from readiness import hbedb

pytestmark = pytest.mark.full


@pytest.fixture(scope="module")
def all_trials():
    return [hbedb.load_trial(t) for t in hbedb.trial_ids()]


def test_counts(all_trials):
    assert len(all_trials) == 1930
    assert len({t.subject for t in all_trials}) == 163
    counts = {c: sum(t.condition == c for t in all_trials) for c in C.CONDITIONS}
    assert counts == {"firm_eo": 489, "firm_ec": 487, "foam_eo": 480, "foam_ec": 474}


def test_samples(all_trials):
    assert all(t.data.shape == (C.N_SAMPLES, C.N_CHANNELS) for t in all_trials)
    assert all(np.isfinite(t.data).all() for t in all_trials)


def test_vertical_force_equals_body_weight(all_trials):
    ratio = np.array([np.mean(t.channel("Fz")) / (t.mass_kg * C.G) for t in all_trials])
    assert 1.00 <= np.median(ratio) <= 1.02
    assert ratio.min() >= 0.94 and ratio.max() <= 1.05


def test_cop_from_moments(all_trials):
    # COPx (cm) = -My/Fz x 100 and COPy = Mx/Fz x 100 numerically (pins COP scale and offset, not only shape)
    for t in all_trials:
        fz = t.channel("Fz")
        assert np.max(np.abs(t.channel("COPx") + t.channel("My") / fz * 100.0)) < 1e-3
        assert np.max(np.abs(t.channel("COPy") - t.channel("Mx") / fz * 100.0)) < 1e-3


def test_fx_sign_and_channel(all_trials):
    # Inverted pendulum: force on the plate Fx is proportional to COP - COM, i.e. to the fast part of COPx.
    # A sign flip makes this negative; swapping in Fy makes it ~0. Every correlation must be finite.
    from scipy.signal import butter, filtfilt
    bb, aa = butter(2, 0.5 / (C.FS_HZ / 2), "high")
    cx, cy = [], []
    for t in all_trials:
        hp = filtfilt(bb, aa, t.channel("COPx"))
        cx.append(np.corrcoef(t.channel("Fx"), hp)[0, 1])
        cy.append(np.corrcoef(t.channel("Fy"), hp)[0, 1])
    cx, cy = np.array(cx), np.array(cy)
    assert np.all(np.isfinite(cx)) and np.all(np.isfinite(cy))
    assert np.median(cx) > 0.8 and np.mean(cx > 0.3) > 0.99
    assert abs(np.median(cy)) < 0.2


def test_matches_independent_decoder(all_trials):
    # Reference decoder: the PhysioNet wfdb package (independent header parsing and calibration)
    wfdb = pytest.importorskip("wfdb", reason="install requirements-dev.txt for the independent decoder check")
    root = str(C.data_dir())
    for t in all_trials:
        rec = wfdb.rdrecord(f"{root}/{t.trial_id}")
        assert tuple(rec.sig_name) == C.CHANNELS
        assert np.allclose(rec.p_signal, t.data, rtol=0, atol=1e-9), t.trial_id


def test_constant_channel_is_rejected_by_gate_logic():
    # Guard for the gate itself: a constant COP channel gives an undefined correlation, which must fail
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(np.ones(100), np.arange(100.0))[0, 1]
    assert not np.isfinite(c)


def test_missing_trials_reported():
    miss = hbedb.missing_trials()
    assert int(miss["missing"].sum()) == 163 * 12 - 1930  # 26
    assert miss["Subject"].nunique() == 5


def test_healthy_flags():
    info = hbedb.load_info()
    per_subj = info.groupby("Subject")[["vestibular"]].first()
    assert int(per_subj["vestibular"].sum()) == 12
