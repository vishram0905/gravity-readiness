# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""The interactive app: user-set inputs drive the decision engine, and every displayed result is the engine's."""
import json
import re

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from readiness import config as C
from readiness import params as P
from readiness import scenarios as SC
from readiness.decision import ConditionInput, decide, decide_confirmed

PEOPLE = json.loads((C.DEMO_DIR / "demo_people.json").read_text())


def _app():
    at = AppTest.from_file(str(C.REPO_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture(scope="module")
def at():
    return _app()


def _card(at):
    m = next(m.value for m in at.markdown if "Recommended action" in m.value)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m)).strip()


def _status(at):
    for s in ("CANNOT ASSESS", "NO-GO", "CAUTION", "GO"):
        if f" {s}" in " " + _card(at).split("Rule")[0]:
            return s


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for kind in ("info", "warning", "success", "error"):
        parts += [x.value for x in getattr(at, kind)]
    return " ".join(parts)


def _rule_table(at):
    return next(d.value for d in at.dataframe if "Result" in d.value.columns)


def _load(at, prefix):
    name = next(n for n in at.selectbox(key="example").options if n.startswith(prefix))
    at.selectbox(key="example").set_value(name).run()
    assert not at.exception, at.exception
    return at


# ---------------------------------------------------------------- examples reproduce the engine's scenario results
def _expected():
    out = {}
    for i, (key, subj) in enumerate(PEOPLE.items()):
        p = SC.load_person(int(subj))
        for name in SC.SCENARIOS:
            r = SC.run_scenario(p, name)
            out[(chr(65 + i), name)] = (r.overall.status, r.go_reached)
    return out


def test_every_example_gives_the_engine_result(at):
    exp = _expected()
    stage = {"baseline day, no landing": "Healthy check", "6 h after landing": "Just landed (6 h)",
             "first GO": "First GO test", "GO not reached": "First GO test",
             "confirmatory session": "Confirmation test 2 h later", "simulated fall": "Fall during test"}
    names = at.selectbox(key="example").options
    assert len(names) == 13
    for n in names:
        at.selectbox(key="example").set_value(n).run()
        assert not at.exception
        scen = next(v for k, v in stage.items() if n.endswith(k))
        assert _status(at) == exp[(n[0], scen)][0], n


def test_young_participant_story(at):
    assert _status(_load(at, "A · younger adult · 6 h")) == "NO-GO"
    assert "Retest in approximately 63 h (≈69 h after landing)" in _card(at)
    assert _status(_load(at, "A · younger adult · 69 h")) == "CAUTION"
    assert "single GO session · confirmatory test required" in _card(at)
    assert _status(_load(at, "A · younger adult · 71 h")) == "GO"
    assert "egress decision by the crew surgeon" in _card(at)
    assert _status(_load(at, "A · younger adult · baseline day, simulated fall")) == "NO-GO"


# ---------------------------------------------------------------- the controls drive the decision
def test_score_below_75_percent_gives_nogo_rule_3():
    at = _load(_app(), "A · younger adult · 71 h")
    b = SC.load_person(PEOPLE["Young healthy"]).baseline("foam_ec")
    at.slider(key="eq_foam_ec").set_value(round(0.70 * b, 1)).run()
    assert _status(at) == "NO-GO" and "Rule 3:" in _card(at) and "USER-ENTERED VALUES" in _card(at)


def test_fall_gives_nogo_whatever_the_score():
    at = _load(_app(), "A · younger adult · 71 h")
    at.slider(key="eq_foam_ec").set_value(100.0).run()
    at.radio(key="out_foam_ec").set_value("fall").run()
    assert _status(at) == "NO-GO" and "Rule 2:" in _card(at)
    assert "Do not leave the vehicle unassisted." in _card(at)


def test_invalid_recording_gives_cannot_assess():
    at = _load(_app(), "A · younger adult · 71 h")
    at.radio(key="out_firm_eo").set_value("invalid").run()
    assert _status(at) == "CANNOT ASSESS" and "Rule 1:" in _card(at)


def test_typical_recovery_button_sets_the_nasa_curve():
    at = _load(_app(), "A · younger adult · 6 h")
    at.slider(key="t").set_value(24.0).run()
    next(b for b in at.button if b.label.startswith("Set scores to typical")).click().run()
    p = SC.load_person(PEOPLE["Young healthy"])
    for c in C.CONDITIONS:
        assert at.slider(key=f"eq_{c}").value == pytest.approx(min(100.0, p.baseline(c) * SC.r_sim(24.0, c, p)), abs=0.06)


@pytest.mark.parametrize("seed", range(6))
def test_random_inputs_match_the_engine(seed):
    """Whatever the user sets, the card and the rule table show the engine's own result."""
    rng = np.random.default_rng(seed)
    at = _app()
    who = list(at.selectbox(key="who").options)[seed % 3]
    at.selectbox(key="who").set_value(who).run()
    p = SC.load_person(list(PEOPLE.values())[seed % 3])
    vals, outs = {}, {}
    for c in C.CONDITIONS:
        vals[c] = round(float(rng.uniform(40, 100)), 1)
        outs[c] = str(rng.choice(["normal"] * 8 + ["fall", "invalid"]))
        at.slider(key=f"eq_{c}").set_value(vals[c])
        at.radio(key=f"out_{c}").set_value(outs[c])
    at.run()
    inputs = {c: ConditionInput([vals[c]] * 3, [outs[c] == "fall", False, False], p.baseline(c),
                                P.noise_q95(p.baseline(c), c, p.age_group), P.p5(c, p.age_group),
                                [outs[c] != "invalid"] * 3, p.baseline_valid(c)) for c in C.CONDITIONS}
    test = decide(inputs, "S2", 16.0)
    assert _status(at) == decide_confirmed([test]).status
    table = _rule_table(at)
    for c in C.CONDITIONS:
        shown = table.loc[table.Condition == C.CONDITION_LABELS[c], "Result"].item()
        assert test.per_condition[c].status in shown


# ---------------------------------------------------------------- rules are visible
def test_rules_are_listed_and_the_applied_rule_is_marked(at):
    _load(at, "A · younger adult · 6 h")
    rules = next(m.value for m in at.markdown if m.value.startswith("1. "))
    assert rules.count("\n") == 6 and "**A score below 75% of baseline yields NO-GO.** *(applied)*" in rules
    assert list(_rule_table(at).columns) == ["Condition", "% of baseline", "1 valid", "2 no fall", "3 ≥ 75%",
                                             "4 ≥ floor", "5 ≥ GO threshold", "Result"]


# ---------------------------------------------------------------- medicine: forecast only
def test_medicine_never_changes_the_test():
    at = _load(_app(), "A · younger adult · 6 h")
    card0, table0 = _card(at).split("Recommended action")[0], _rule_table(at)
    at.select_slider(key="dose").set_value(25.0).run()
    assert _card(at).split("Recommended action")[0] == card0 and _rule_table(at).equals(table0)
    text = _text(at)
    assert "Pharmacological scenario active" in text and "(unvalidated scenario)" in text
    assert "intravenous promethazine" in text and "with the planned dose" in _card(at)
    h0, h1 = map(float, re.search(r"Foam, eyes open reaches its GO threshold at approximately ([0-9.]+) h after landing "
                                  r"without the dose and ([0-9.]+) h with it", text).groups())
    assert h0 == 63.6 and h1 > h0


def test_medicine_not_offered_without_landing_or_with_a_fall():
    at = _load(_app(), "A · younger adult · baseline day, no landing")
    assert "Not applicable to the present inputs" in _text(at)
    at = _load(_app(), "A · younger adult · 6 h")
    at.radio(key="out_foam_ec").set_value("fall").run()
    assert "Not applicable to the present inputs" in _text(at) and "Estimated retest time" not in [h.value for h in at.subheader]


# ---------------------------------------------------------------- forecast
def test_unreachable_threshold_gives_no_retest_time(at):
    _load(at, "C · low baseline balance · 6 h")
    assert "No retest time can be estimated" in _card(at) and "Refer to the crew surgeon." in _card(at)


def test_no_chart_clips_its_curves():
    for prefix in ("A · younger adult · 6 h", "B · older adult · 6 h", "C · low baseline balance · 240 h"):
        at = _load(_app(), prefix)
        for c in at.get("plotly_chart"):
            spec = json.loads(c.proto.spec)
            for tr in spec["data"]:  # each trace against the y-axis of its own panel
                ax = spec["layout"].get("yaxis" + tr.get("yaxis", "y")[1:], {})
                rng = ax.get("range")
                if not rng or ax.get("tickvals"):
                    continue
                ys = [y for y in tr.get("y", []) if y is not None]
                if ys:
                    assert rng[0] - 1e-9 <= min(ys) and max(ys) <= rng[1] + 1e-9


# ---------------------------------------------------------------- framing and context
def test_safety_qualifiers_on_the_first_screen(at):
    _load(at, "A · younger adult · 6 h")
    text = _text(at)
    assert "does not constitute clearance" in text and "not validated for egress" in text
    assert "A retest, not this estimate, decides GO" in _card(at) and "SIMULATED EXAMPLE" in _card(at)
    assert "Not a medical device" in text


def test_osdr_panel_is_context_only(at):
    assert "Additional crew data (context)" in [t.label for t in at.tabs]
    text = _text(at)
    assert "do not enter the balance decision" in text and "OSD-569" in text and "OSD-575" in text


def test_wood_r0_described_as_model_estimate():
    text = (C.REPO_ROOT / "app.py").read_text() + (C.REPO_ROOT / "README.md").read_text()
    assert "13.2 on landing day" not in text and text.count("13.2 at landing") >= 2


def test_safety_instructions_keep_their_force():
    from readiness import medicine as M
    assert "Stand only with assistance." in M.WARN_ORTHOSTATIC


# ---------------------------------------------------------------- independent review of the interactive model (15 Nov)
def test_recovery_button_keeps_a_fall():
    """Review C01: generating scores never erases a fall or an invalid recording."""
    at = _load(_app(), "A · younger adult · 71 h")
    at.radio(key="out_foam_ec").set_value("fall").run()
    at.radio(key="out_firm_eo").set_value("invalid").run()
    next(b for b in at.button if b.label.startswith("Set scores to typical")).click().run()
    assert at.radio(key="out_foam_ec").value == "fall" and at.radio(key="out_firm_eo").value == "invalid"
    assert _status(at) == "CANNOT ASSESS"
    at.radio(key="out_firm_eo").set_value("normal").run()
    assert _status(at) == "NO-GO" and "Rule 2:" in _card(at)


def test_generated_scores_are_labelled_simulated():
    """Review C02: scores from the recovery model are labelled as simulated, edits as user-entered."""
    at = _load(_app(), "A · younger adult · 6 h")
    at.slider(key="t").set_value(30.0).run()
    next(b for b in at.button if b.label.startswith("Set scores to typical")).click().run()
    assert "SIMULATED: TYPICAL NASA RECOVERY" in _card(at)
    at.slider(key="eq_firm_eo").set_value(50.0).run()
    assert "USER-ENTERED VALUES" in _card(at)


def test_score_above_baseline_is_not_called_unreachable():
    """Review C03: a condition already above its GO threshold is reported as passed, not unreachable, and its
    current point lies inside the chart."""
    at = _app()
    at.selectbox(key="who").set_value(at.selectbox(key="who").options[1]).run()
    for c, v in zip(C.CONDITIONS, (85.0, 95.0, 95.0, 95.0)):
        at.slider(key=f"eq_{c}").set_value(v)
    at.run()
    assert not at.exception
    text = _text(at)
    assert "Foam, eyes open is not predicted" not in text and "foam, eyes open · foam, eyes closed) are already at or above" in text
    for ch in at.get("plotly_chart"):
        spec = json.loads(ch.proto.spec)
        if "Retest" in spec["layout"].get("title", {}).get("text", "") or "retest" in spec["layout"].get("title", {}).get("text", ""):
            top = spec["layout"]["yaxis"]["range"][1]
            pts = [y for tr in spec["data"] if tr.get("mode") == "markers" for y in tr["y"]]
            assert all(y <= top for y in pts)


def test_planned_dose_cannot_shorten_the_retest_estimate():
    """Review C04: a condition that passes now but is lowered by a planned dose still enters the retest estimate."""
    at = _load(_app(), "A · younger adult · 6 h")
    for c, v in zip(C.CONDITIONS, (90.7, 95.0, 82.0, 85.0)):
        at.slider(key=f"eq_{c}").set_value(v)
    at.run()
    at.select_slider(key="dose").set_value(50.0)
    at.radio(key="route").set_value("im")
    at.run()
    assert not at.exception
    card, text = _card(at), _text(at)
    h_card = float(re.search(r"≈([0-9.]+) h after landing", card).group(1))
    h_all = float(re.search(r"All four conditions reach their GO thresholds at approximately [0-9.]+ h after landing "
                            r"without the dose and ([0-9.]+) h with it", text).group(1))
    assert abs(h_card - h_all) <= 1.0, (card, h_all)


def _c04_case(foam_eo_first=90.7):
    at = _load(_app(), "A · younger adult · 6 h")
    for c, v in zip(C.CONDITIONS, (foam_eo_first, 95.0, 82.0, 85.0)):
        at.slider(key=f"eq_{c}").set_value(v)
    at.run()
    at.select_slider(key="dose").set_value(50.0)
    at.radio(key="route").set_value("im")
    at.run()
    assert not at.exception
    return at


def test_planned_dose_is_kept_when_the_session_becomes_go():
    """Review r2 C01: a GO session keeps the planned dose, and the confirmatory test is not advised before the
    dose-adjusted time."""
    at = _c04_case()
    h_before = float(re.search(r"≈([0-9.]+) h after landing", _card(at)).group(1))
    at.slider(key="eq_firm_eo").set_value(90.8).run()
    card = _card(at)
    assert _status(at) == "CAUTION" and "single GO session" in card
    assert "Pharmacological scenario active" in _text(at)
    h = float(re.search(r"below the GO threshold until approximately ([0-9.]+) h after landing", card).group(1))
    assert abs(h - h_before) <= 1.0 and "should not precede that time" in card
    assert "(from 8 h after landing)" not in card
    at.session_state["history"] = [{"t": 4.0, "trials": {c: [95.0] * 3 for c in C.CONDITIONS},
                                    "out": {c: "normal" for c in C.CONDITIONS}}]
    at.run()
    next(b for b in at.button if b.label == "Record this session").click().run()
    assert _status(at) == "GO" and "Reassessment after the dose is advised" in _card(at)


def test_go_session_without_dose_shows_no_retest_chart():
    at = _load(_app(), "A · younger adult · 69 h")
    assert "Estimated retest time" not in [h.value for h in at.subheader]


def test_retest_shading_matches_the_caption_range():
    """Review r2 C02: the shaded range is the overall range given in the caption."""
    at = _load(_app(), "A · younger adult · 6 h")
    lo, hi = map(float, re.search(r"Range ([0-9]+)–([0-9]+) h after this session", _text(at)).groups())
    spec = next(json.loads(c.proto.spec) for c in at.get("plotly_chart")
                if "Retest estimate" in json.loads(c.proto.spec)["layout"].get("title", {}).get("text", ""))
    rect = next(sh for sh in spec["layout"]["shapes"] if sh["type"] == "rect")
    assert round(rect["x0"] - 6.0) == lo and round(rect["x1"] - 6.0) == hi


# ---------------------------------------------------------------- judge-perspective improvements (15 Nov)
def test_story_examples_come_first(at):
    names = at.selectbox(key="example").options
    assert names[:4] == ["A · younger adult · 6 h after landing", "A · younger adult · 69 h after landing, first GO",
                         "A · younger adult · 71 h after landing, confirmatory session",
                         "A · younger adult · baseline day, simulated fall"]


def test_problem_measurement_and_nasa_data_are_stated(at):
    _load(at, "A · younger adult · 6 h")
    text = _text(at)
    assert "13.2 at landing against 74.9 before flight" in text and "no ground team" in text
    assert "NASA data in this result:" in text and "Jain et al. 2010" in text and "Table III, n = 37" in text


def test_population_floor_is_shown(at):
    _load(at, "A · younger adult · 6 h")
    table = _rule_table(at)
    p = SC.load_person(PEOPLE["Young healthy"])
    shown = table.loc[table.Condition == "firm floor, eyes open", "4 ≥ floor"].item()
    assert f"({P.p5('firm_eo', p.age_group) / p.baseline('firm_eo'):.0%})" in shown


def test_typical_crew_marker_follows_the_nasa_curve(at):
    _load(at, "A · younger adult · 6 h")
    from readiness import forecast as F
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    rings = [tr["x"][0] for tr in spec["data"] if "typical ISS crew" in tr.get("hovertemplate", "")]
    exp = sorted(F.recovery(6.0, F.wood(c)["D"], F.wood(c)["tau"]) for c in C.CONDITIONS)
    assert sorted(rings) == pytest.approx(exp)


def test_measurement_tab_reproduces_the_baseline_score(at):
    """The force-plate trace and its EQ are the participant's recorded first baseline trial."""
    _load(at, "A · younger adult · 6 h")
    p = SC.load_person(PEOPLE["Young healthy"])
    line = next(m.value for m in at.markdown if m.value.startswith("θ = atan("))
    eq = float(re.search(r"= \*\*([0-9.]+)\*\*", line).group(1))
    assert eq == pytest.approx(p.w1["foam_ec"][0], abs=0.05)



# ---------------------------------------------------------------- recorded sessions and three-trial input (judge review)
def test_two_session_rule_uses_recorded_sessions():
    at = _load(_app(), "A · younger adult · 69 h")
    assert _status(at) == "CAUTION"
    next(b for b in at.button if b.label == "Record this session").click().run()
    # Judge review r2 C01: recording one GO session never confirms itself; the next session must be entered.
    assert at.slider(key="t").value == 71.0 and _status(at) == "CAUTION" and "single GO session" in _card(at)
    assert "Advancing the time alone does not create a session" in _text(at)
    next(b for b in at.button if b.label.startswith("Set scores to typical")).click().run()  # the 71-h session
    # Judge review r2 C01 (re-check): an entered confirmatory session does not confirm until it is recorded.
    assert _status(at) == "CAUTION" and "Record this session to complete the confirmatory test" in _card(at)
    assert "69 h GO (simulated) · 71 h GO (current, simulated)" in _text(at)
    at.slider(key="t").set_value(70.0).run()  # less than 2 h after the recorded session
    assert _status(at) == "CAUTION" and "Record this session to complete" not in _card(at)
    at.slider(key="t").set_value(69.0).run()  # not after the recorded session: order cannot be established
    assert _status(at) == "CANNOT ASSESS" and "Rule 7:" in _card(at)
    at.slider(key="t").set_value(71.0).run()
    next(b for b in at.button if b.label == "Record this session").click().run()
    assert _status(at) == "GO" and "69 h GO (simulated) · 71 h GO (simulated)" in _text(at)


def test_an_incidental_edit_never_confirms():
    """Judge review r2 C01 (re-check): a one-value edit of carried-over scores is not a recorded session."""
    at = _load(_app(), "A · younger adult · 69 h")
    at.toggle(key="trials_mode").set_value(True).run()
    next(b for b in at.button if b.label == "Record this session").click().run()
    at.number_input(key="tr_firm_eo_0").set_value(at.number_input(key="tr_firm_eo_0").value + 0.1).run()
    assert _status(at) == "CAUTION" and len(at.session_state["history"]) == 1
    at.radio(key="out_foam_ec").set_value("fall").run()
    at.radio(key="out_foam_ec").set_value("normal").run()
    assert _status(at) == "CAUTION"


def test_recorded_go_never_lifts_a_failing_session():
    at = _load(_app(), "A · younger adult · 69 h")
    next(b for b in at.button if b.label == "Record this session").click().run()
    b = SC.load_person(PEOPLE["Young healthy"]).baseline("foam_ec")
    at.slider(key="eq_foam_ec").set_value(round(0.6 * b, 1)).run()
    assert _status(at) == "NO-GO"


def test_confirmatory_example_carries_its_first_session(at):
    _load(at, "A · younger adult · 71 h")
    assert [h["t"] for h in at.session_state["history"]] == [69.0, 71.0]
    assert "69 h GO (simulated) · 71 h GO (simulated)" in _text(at) and _status(at) == "GO"


def test_three_trial_mode_uses_the_recorded_trials():
    at = _load(_app(), "A · younger adult · 6 h")
    at.toggle(key="trials_mode").set_value(True).run()
    p = SC.load_person(PEOPLE["Young healthy"])
    r = SC.run_scenario(p, "Just landed (6 h)")
    for c in C.CONDITIONS:
        assert [at.number_input(key=f"tr_{c}_{k}").value for k in range(3)] == pytest.approx(r.test.inputs[c].trial_eqs)
    assert _status(at) == "NO-GO" and "SIMULATED EXAMPLE" in _card(at)
    at.number_input(key="tr_foam_ec_0").set_value(100.0).run()
    at.number_input(key="tr_foam_ec_1").set_value(100.0).run()  # median now 100: no longer below 75%
    table = _rule_table(at)
    assert "NO-GO" not in table.loc[table.Condition == "foam, eyes closed", "Result"].item()


def test_changing_the_crew_member_clears_the_history(at):
    _load(at, "A · younger adult · 71 h")
    at.selectbox(key="who").set_value(at.selectbox(key="who").options[1]).run()
    assert at.session_state["history"] == []


def test_provenance_line_follows_the_inputs():
    at = _load(_app(), "A · younger adult · 6 h")
    at.radio(key="out_foam_ec").set_value("fall").run()
    line = next(m.value for m in at.markdown if m.value.startswith("**NASA data in this result"))
    assert "Retest time" not in line and "applied here to a centre-of-mass estimate" not in line



def test_edited_session_keeps_simulated_history_visible():
    """Judge review r2 C02: an assessment resting on a simulated earlier session says so after edits."""
    at = _load(_app(), "A · younger adult · 71 h")
    for c in C.CONDITIONS:
        at.slider(key=f"eq_{c}").set_value(100.0)
    at.run()
    assert "USER-ENTERED VALUES · SIMULATED SESSION IN HISTORY" in _card(at) and _status(at) == "CAUTION"
    assert "69 h GO (simulated)" in _text(at)


def test_record_is_blocked_until_the_next_session_is_entered():
    at = _load(_app(), "A · younger adult · 69 h")
    next(b for b in at.button if b.label == "Record this session").click().run()
    assert next(b for b in at.button if b.label == "Record this session").disabled


# ---------------------------------------------------------------- independent judge review r4 (15 Nov)
def _set_all(at, t, scores, outcome="normal"):
    at.slider(key="t").set_value(t)
    for c, v in zip(C.CONDITIONS, scores):
        at.slider(key=f"eq_{c}").set_value(v)
        at.radio(key=f"out_{c}").set_value(outcome)
    at.run()
    assert not at.exception, at.exception
    return at


def test_confirmation_beyond_the_time_range_is_stated_not_offered():
    """C01: at the end of the supported range the app neither promises an unreachable session nor misdiagnoses it."""
    at = _load(_app(), "A · younger adult · 6 h")
    _set_all(at, 240.0, [100.0] * 4)
    assert "beyond the supported range of 240 h" in _card(at) and "from 242 h" not in _card(at)
    _set_all(at, 238.0, [100.0] * 4)
    assert "from 240 h after landing" in _card(at)
    rec = lambda: next(b for b in at.button if b.label == "Record this session")
    rec().click().run()
    assert at.session_state["t"] == 240.0
    at.slider(key="eq_firm_eo").set_value(99.9).run()
    assert _status(at) == "CAUTION" and "Record this session" in _card(at) and not rec().disabled
    rec().click().run()  # Judge r5 C01: the advertised 240 h confirmation can be completed
    assert _status(at) == "GO" and [h["t"] for h in at.session_state["history"]] == [238.0, 240.0]
    assert "No further session can be recorded" in _text(at) and rec().disabled
    at.slider(key="eq_firm_eo").set_value(99.8).run()  # nothing later than 240 h is possible
    assert _status(at) == "CANNOT ASSESS" and "Clear the session history" in _card(at)
    at = _load(_app(), "A · younger adult · 69 h")
    rec = lambda: next(b for b in at.button if b.label == "Record this session")
    rec().click().run()
    at.slider(key="t").set_value(60.0)
    at.slider(key="eq_firm_eo").set_value(99.0)
    at.run()  # a time before the last recorded session
    assert _status(at) == "CANNOT ASSESS" and "at least 2 h after the last recorded session" in _card(at)


def test_late_forecast_curve_runs_forward_through_the_estimate():
    """C02: the retest chart starts at the measurement and reaches past the estimated crossing."""
    at = _load(_app(), "A · younger adult · 6 h")
    _set_all(at, 240.0, [95.0, 95.0, 95.0, 0.0])
    mid = float(re.search(r"≈([0-9]+) h after landing", _card(at)).group(1))
    spec = next(json.loads(c.proto.spec) for c in at.get("plotly_chart")
                if "Retest estimate" in json.loads(c.proto.spec)["layout"].get("title", {}).get("text", ""))
    curve = spec["data"][0]
    assert min(curve["x"]) == pytest.approx(240.0) and max(curve["x"]) > mid
    assert min(curve["y"]) >= -1e-9  # starts at the measured score (0), never extrapolated below it


def test_loading_an_example_resets_the_medicine_scenario():
    """C03: an example reproduces its own recommendation whatever medicine was set before."""
    fresh = _card(_load(_app(), "A · younger adult · 69 h"))
    at = _load(_app(), "A · younger adult · 6 h")
    at.select_slider(key="dose").set_value(50.0).run()
    at.radio(key="route").set_value(at.radio(key="route").options[-1]).run()
    _load(at, "A · younger adult · 69 h")
    assert at.session_state["dose"] == 0.0 and _card(at) == fresh


def test_recovery_comparison_with_typical_nasa_crew():
    at = _load(_app(), "A · younger adult · 71 h")
    assert "Recovery compared with typical NASA crew" in [s.value for s in at.subheader]
    assert "Difference from typical ISS crew (recorded 71 h)" in _text(at)
    assert "does not enter the decision" in _text(at)
    at = _load(_app(), "A · younger adult · baseline day, no")
    assert "Recovery compared with typical NASA crew" not in [s.value for s in at.subheader]


def test_recovery_comparison_ignores_invalid_and_out_of_order_sessions():
    """Judge r5 C02: only valid measurements in time order are compared with the NASA curve."""
    at = _load(_app(), "A · younger adult · 6 h")
    _set_all(at, 6.0, [100.0] * 4, outcome="invalid")
    assert _status(at) == "CANNOT ASSESS"
    assert _text(at).count("no valid recording") == 4
    at = _load(_app(), "A · younger adult · 71 h")
    at.slider(key="eq_firm_eo").set_value(99.9)
    at.slider(key="t").set_value(6.0)
    at.run()
    assert "(recorded 71 h)" in _text(at) and "precedes the last recorded session and is not plotted" in _text(at)
