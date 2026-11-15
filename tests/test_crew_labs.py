# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""NASA OSDR Inspiration4 subset: shape, provenance against the released files, and the own-preflight arithmetic."""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from readiness import crew_labs as L

SRC = Path(os.environ.get("OSDR_I4_DIR", "")) if os.environ.get("OSDR_I4_DIR") else None


def test_subset_shape():
    df = L.load()
    assert len(df) == len(L.ANALYTES) * 4 * len(L.TIMEPOINTS)  # 7 analytes x 4 crew x 7 time points
    assert set(df.subject) == {"C001", "C002", "C003", "C004"} and set(df.timepoint) == set(L.TIMEPOINTS)
    assert "SEX" not in df.columns and "sex" not in df.columns
    assert df.value.notna().all()


def test_share_is_value_over_own_preflight_mean():
    r = L.relative_to_preflight(L.load())
    row = r[(r.analyte == "SODIUM") & (r.subject == "C001") & (r.timepoint == "R+1")].iloc[0]
    pre = r[(r.analyte == "SODIUM") & (r.subject == "C001") & r.timepoint.isin(L.PREFLIGHT)]["value"]
    assert row.value == 138 and np.isclose(row.preflight_mean, (141 + 139 + 142) / 3)
    assert np.isclose(row.share, 138 / ((141 + 139 + 142) / 3))
    assert np.allclose(r[r.timepoint.isin(L.PREFLIGHT)].groupby(["analyte", "subject"])["share"].mean(), 1.0)


@pytest.mark.skipif(SRC is None, reason="set OSDR_I4_DIR to the folder with the released OSDR files")
def test_every_value_matches_the_released_osdr_files():
    rel = {}
    for f, tcol in (("LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv", "TEST_DATE"),
                    ("LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv", "timepoint")):
        for r in csv.DictReader((SRC / f).open(encoding="utf-8-sig")):
            name = r["ANALYTE"].replace(" (FEMALE)", "")
            if name in L.ANALYTES:
                rel[(r["SUBJECT_ID"], name, r[tcol])] = float(r["VALUE"])
    df = L.load()
    for r in df.itertuples():
        assert rel[(r.subject, r.analyte, r.timepoint)] == r.value
