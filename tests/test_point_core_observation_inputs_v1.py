"""验证核心观察预算在真实月首收盘更新，且后续资料不能改变此前状态。"""
import numpy as np
import pandas as pd
import pytest

from research import point_core_observation_inputs_v1 as study


def sample():
    dates = pd.bdate_range("2024-01-02", "2024-03-05")
    t = np.arange(len(dates))
    returns = np.column_stack([.004*np.sin(t), .008*np.cos(t*.7)])
    states = np.column_stack([np.ones(len(t)), np.zeros(len(t))])
    return dates, returns, states


@pytest.mark.parametrize("kind", ["variance", "joint"])
def test_month_first_last_observation_is_calculated_and_prefix_is_stable(kind):
    dates, returns, states = sample()
    index = int(np.flatnonzero(dates == pd.Timestamp("2024-03-01"))[0])
    if kind == "variance":
        short = study.observe_variance_budget(dates[:index+1], returns[:index+1], states[:index+1], 1, window=5)
        full = study.observe_variance_budget(dates, returns, states, 1, window=5)
        assert np.isfinite(short.target.iloc[-1])
    else:
        short = study.observe_joint_budget(dates[:index+1], returns[:index+1], 1, window=5)
        full = study.observe_joint_budget(dates, returns, 1, window=5)
        assert np.isfinite(short.downside_budget.iloc[-1])
    assert short.risk_update_scheduled.iloc[-1]
    assert short.last_successful_risk_origin.iloc[-1] == dates[index]
    pd.testing.assert_frame_equal(short, full.iloc[:index+1].reset_index(drop=True))


def test_unknown_parent_stays_unknown_even_when_weight_zero():
    result = study.weighted_targets([np.nan, .2, .4], [.5, np.nan, .8], [0., 1., .25])
    assert np.isnan(result[:2]).all()
    assert result[2] == pytest.approx(.7)


def test_reference_intent_rejects_shifted_execution_date():
    dates, _, _ = sample()
    dates = dates[:4]
    data = pd.DataFrame({"date": dates})
    decisions = pd.DataFrame({"origin": dates, "execution_date": dates, "origin_index": np.arange(4), "reference_weight": 1.})
    with pytest.raises(ValueError, match="执行时钟"):
        study.intent_array(data, decisions, dates[1], dates[-1]+pd.offsets.BDay(1))


def test_terminal_open_return_cannot_enter_observation_budget():
    dates, _, _ = sample()
    dates = dates[:4]
    data = pd.DataFrame({"date": dates})
    ledger = pd.DataFrame({"date": dates[1:], "net_return": 0., "mark_clock": ["CLOSE", "CLOSE", "OPEN_TERMINAL"]})
    with pytest.raises(ValueError, match="人工终点"):
        study.return_array(data, ledger, dates[1])
