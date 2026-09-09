"""仅合成数据：验证无未来泄漏、HC3、成对安慰剂及来源停止线。"""
import numpy as np
import pandas as pd
import pytest

from research.nbs_v2_common import ContractError
from research.nbs_v2_g2_engine import (
    StatisticalNoView, admitted_loader, bootstrap_slopes, construct_labels, g2_statistics,
    make_plan, ols_hc3, prequential,
)


def synthetic(n=86):
    rng = np.random.default_rng(2005)
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = np.maximum(rng.normal(size=n), 0.)
    return pd.DataFrame({"event_id": [f"synthetic_{i}" for i in range(n)],
        "observation_date": dates.strftime("%Y-%m-%d"),
        "origin_at": dates.strftime("%Y-%m-%dT10:05:00+08:00"),
        "maturity_at": dates.strftime("%Y-%m-%dT14:55:00+08:00"),
        "era": ["TRAINING_ORIGIN"] * 36 + ["ERA_1"] * 17 + ["ERA_2"] * 17 + ["ERA_3"] * (n - 70),
        "X": x, "Y": .003 * x + rng.normal(0, .0003, n)})


def test_g0_failure_never_calls_outcome_loader():
    calls = []
    source = {"state": "BLOCKED_REFETCHED_DATA_ORIGINAL_V2_CONTRACT_FAILED", "source_pass": False,
              "checks": {"all_bar_vwap_inside_low_high": False}, "source_hard_gates_changed": False}
    with pytest.raises(ContractError):
        admitted_loader(source, lambda: calls.append("READ_Y"))
    assert calls == []


def test_false_pass_label_or_changed_gate_cannot_open_loader():
    for changed, passed in [(False, False), (True, True)]:
        source = {"state": "PASS_REFETCHED_DATA_ORIGINAL_V2_CONTRACT", "source_pass": True,
                  "checks": {"bar_gate": passed}, "source_hard_gates_changed": changed}
        with pytest.raises(ContractError):
            admitted_loader(source, lambda: pytest.fail("不得读取"))


def test_hc3_matches_delete_one_coefficient_perturbations():
    x = np.array([0., 0., 1., 2., 3., 5.])
    y = np.array([.001, -.002, .003, .008, .006, .020])
    coef, cov = ols_hc3(x, y)
    design = np.column_stack([np.ones(len(x)), x])
    changes = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        deleted = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        changes.append(deleted - coef)
    changes = np.array(changes)
    np.testing.assert_allclose(cov, changes.T @ changes, rtol=1e-10, atol=1e-15)


def test_current_and_future_targets_do_not_change_current_forecast():
    original = synthetic()
    before = prequential(original)
    changed = original.copy()
    changed.loc[36:, "Y"] += 100
    after = prequential(changed)
    for field in ["alpha", "beta", "b0", "b1", "mean_se_hc3", "lcb"]:
        assert before.iloc[0][field] == after.iloc[0][field]
    assert before.iloc[0].training_events == 36
    assert before.iloc[-1].training_events == 85


def test_future_maturity_in_training_stops():
    data = synthetic()
    data.loc[0, "maturity_at"] = "2030-01-01T14:55:00+08:00"
    with pytest.raises(ContractError):
        prequential(data)


def test_constant_feature_is_no_view_no_direction_rescue():
    with pytest.raises(StatisticalNoView):
        ols_hc3(np.zeros(36), np.arange(36))
    coef, _ = ols_hc3(np.arange(36), -np.arange(36) + np.sin(np.arange(36)))
    assert coef[1] < 0


def test_degenerate_bootstrap_remains_nan_instead_of_redraw():
    indices = np.array([[0, 0, 0], [0, 1, 2]])
    slopes = bootstrap_slopes(np.array([0., 1., 2.]), np.array([0., 2., 4.]), indices)
    assert np.isnan(slopes[0])
    assert slopes[1] == 2


def test_identical_placebos_cannot_be_declared_specific_mechanism():
    prediction = prequential(synthetic())
    result, draws = g2_statistics({"MAIN": prediction, "A": prediction.copy(), "B": prediction.copy()}, 10000, 5103001000)
    assert not result["g2_pass"]
    assert not result["gates"]["stronger_than_placebo_A"]
    assert not result["gates"]["stronger_than_placebo_B"]
    np.testing.assert_array_equal(draws.MAIN, draws.A)
    np.testing.assert_array_equal(draws.MAIN, draws.B)


def plan_fixture():
    days = pd.bdate_range("2010-01-01", periods=550)
    ledger = pd.DataFrame({"event_id": [f"test_{i}" for i in range(88)],
        "scheduled_date": days[100:540:5], "final_event_eligibility": True,
        "model_pre_return_eligibility": [False, False] + [True] * 86,
        "model_event_ordinal": [np.nan, np.nan] + list(range(1, 87)),
        "era": ["NOT_MODEL_ELIGIBLE"] * 2 + ["TRAINING_ORIGIN"] * 36
               + ["ERA_1"] * 17 + ["ERA_2"] * 17 + ["ERA_3"] * 16})
    return ledger, days


def test_date_only_plan_excludes_entire_event_family_and_never_reuses_placebo():
    ledger, days = plan_fixture()
    plan = make_plan(ledger, days)
    assert len(plan) == 86
    assert plan.placebo_a_date.nunique() == 86
    family = set(pd.to_datetime(ledger.scheduled_date))
    for row in plan.itertuples():
        for text, limit in [(row.main_and_b_baseline_dates, row.event_date),
                            (row.placebo_a_baseline_dates, row.placebo_a_date)]:
            baseline = pd.to_datetime(text.split("|"))
            assert len(baseline) == 60
            assert (baseline < pd.Timestamp(limit)).all()
            assert not set(baseline) & family
        assert pd.Timestamp(row.placebo_a_date) < pd.Timestamp(row.event_date)


def test_synthetic_minutes_to_three_causal_prequential_arms():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    source = yaml.safe_load((root / "config/510300_stk_mins_source_admission_v2.yaml").read_text("utf-8"))
    model = yaml.safe_load((root / "config/510300_nbs_1000_negative_information_drift_v2.yaml").read_text("utf-8"))
    ledger, days = plan_fixture()
    plan = make_plan(ledger, days)
    windows = dict(source["windows"])
    windows.update({f"shift_{n}": model["placebos"]["B"][f"{n}_labels"] for n in ["pre", "reaction", "entry", "exit"]})
    labels = sorted({label for values in windows.values() for label in values})
    records = []
    for i, day in enumerate(days):
        for j, label in enumerate(labels):
            price = 4 * np.exp(.0005 * np.sin(i / 3) * j / len(labels) + .0002 * np.cos(i / 5 + j / 7))
            records.append({"ts_code": "510300.SH", "trade_time": f"{day:%Y-%m-%d} {label}",
                            "vol": 100., "amount": 100 * price, "low": price - .001, "high": price + .001})
    arms = construct_labels(pd.DataFrame(records), plan, source, model)
    assert set(arms) == {"MAIN", "A", "B"}
    for arm, frame in arms.items():
        assert len(frame) == 86
        assert np.isfinite(frame[["X", "Y"]]).all().all()
        assert (frame.X >= 0).all()
        predictions = prequential(frame)
        assert len(predictions) == 50
        assert predictions.training_events.tolist() == list(range(36, 86))
        assert (pd.to_datetime(predictions.training_last_maturity) < pd.to_datetime(predictions.origin_at)).all()
