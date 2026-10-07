"""共同评分进入实际账户前的边界与完整账户门检验。"""
import numpy as np
import pandas as pd
import pytest

from research.all_factor_macro_earnings_account_v1 import gate, signals


def test_unknown_scores_cannot_request_entry_or_change_calendar():
    dates = pd.bdate_range("2024-09-23", periods=3)
    data = pd.DataFrame({"date": dates, "atr14": [.1, .1, .1]})
    prediction = pd.DataFrame({"date": dates, "policy": "P", "status": "NO_VIEW_NO_MODEL",
                               "candidate_quality_pass": [False, False, False]})
    result = signals(data, prediction, "P")
    assert not result.entry_event.any()
    prediction.loc[1, "candidate_quality_pass"] = True
    with pytest.raises(ValueError, match="未知评分"):
        signals(data, prediction, "P")
    prediction.loc[1, "candidate_quality_pass"] = False
    prediction.loc[1, "date"] = pd.Timestamp("2024-09-30")
    with pytest.raises(ValueError, match="日历"):
        signals(data, prediction, "P")


def test_gate_requires_both_account_improvements_and_actual_net_quality():
    controls = [{"net_cagr": .04, "net_sharpe": 1.2}, {"net_cagr": .05, "net_sharpe": 1.3}]
    candidate = {"net_cagr": .06, "net_sharpe": 1.4, "p_times_b": 1.1,
                 "standard_expectancy_loss_units": .7, "max_drawdown": .08}
    assert gate(candidate, controls)
    for key, value in [("p_times_b", 1.), ("net_cagr", .049), ("net_sharpe", 1.2),
                       ("standard_expectancy_loss_units", 0.), ("max_drawdown", .11), ("p_times_b", np.nan)]:
        assert not gate({**candidate, key: value}, controls)
