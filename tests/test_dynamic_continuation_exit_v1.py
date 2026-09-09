"""核对动态收益的单位、权益边界、终点和同周期连接。"""
import numpy as np
import pandas as pd
import pytest

from research.dynamic_continuation_exit_v1 import make_target, successor_positions, transition_samples
from research.learned_cycle_exit_v1 import FEATURES, continuation_label, training_rows


def fixture_data():
    data = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=6), "open": [10., 10.2, 9.8, 10.4, 10.1, 10.3]})
    dividends = pd.DataFrame({"record_date": [data.date.iloc[0], data.date.iloc[2], data.date.iloc[4]],
                              "cash_dividend_per_share": [.1, .2, .3]})
    rows = []
    for i in range(3):
        rows.append({"cycle_id": 1, "origin_index": i, "early_exit_index": i + 1, "exit_index": 4,
                     "mature_date": data.date.iloc[4], "reference_quantity": 100, "target": 0., "extra_dividend_cny": 0.,
                     **{k: .01 for k in FEATURES}})
    cfg = {"tick": .001, "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}}}
    return data, dividends, pd.DataFrame(rows), cfg


def test_step_returns_telescope_in_same_money_unit():
    data, div, samples, cfg = fixture_data()
    rows = transition_samples(data, div, samples, cfg)
    combined = sum(float(r.one_step_target) * data.open.iloc[int(r.early_exit_index)] / data.open.iloc[1] for r in rows.itertuples())
    direct, rights = continuation_label(data, div, 100, 1, 4, cfg["costs"]["BASE"], cfg["tick"])
    assert combined == pytest.approx(direct, abs=1e-12)
    assert rights == 20.
    assert rows.extra_dividend_cny.sum() == 20.


def test_successor_is_local_and_terminal_does_not_bootstrap():
    data, div, samples, cfg = fixture_data()
    rows = transition_samples(data, div, samples, cfg)
    positions = successor_positions(rows)
    assert positions.tolist() == [1, 2, -1]
    pred = np.array([100., .1, -.2])
    target = make_target(rows, positions, pred)
    assert target[0] == pytest.approx(rows.one_step_target.iloc[0] + .1 * data.open.iloc[2] / data.open.iloc[1])
    assert target[1] == rows.one_step_target.iloc[1]
    assert target[2] == rows.one_step_target.iloc[2]


def test_missing_mid_cycle_state_is_rejected():
    data, div, samples, cfg = fixture_data()
    with pytest.raises((AssertionError, ValueError, RuntimeError), match="不连续"):
        transition_samples(data, div, samples.drop(index=1), cfg)


def test_missing_successor_cannot_be_borrowed_from_another_cycle():
    rows = pd.DataFrame({"cycle_id": [1, 2], "origin_index": [4, 5], "successor_origin_index": [5, -1]})
    with pytest.raises((AssertionError, ValueError, RuntimeError), match="后继状态"):
        successor_positions(rows)


def test_short_step_label_does_not_admit_unfinished_cycle():
    data, div, samples, cfg = fixture_data()
    rows = transition_samples(data, div, samples, cfg)
    early, ids = training_rows(rows, 3, {"recent_cycles": 20})
    assert len(early) == 0 and ids == []
    mature, ids = training_rows(rows, 4, {"recent_cycles": 20})
    assert len(mature) == 3 and ids == [1]
    assert mature.sample_weight.sum() == pytest.approx(1.)
