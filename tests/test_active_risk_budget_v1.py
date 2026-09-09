"""检验条件风险的实际敞口日、数学最小样本及不使用未来。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.active_risk_budget_inputs_v1 import active_budget_frame
from research.two_policy_risk_budget_inputs_v1 import budget_frame


def fixture():
    dates = pd.bdate_range("2020-01-27", periods=35)
    returns = np.zeros((35, 2))
    states = np.tile([1., 0.], (35, 1))
    active = np.zeros((35, 2))
    returns[1:6] = [[-.02, -.01], [0., .01], [.02, -.01], [0., .01], [0., 0.]]
    active[1:6] = [[1, 1], [0, 1], [1, 1], [0, 1], [0, 0]]
    return dates, returns, states, active


def test_conditional_variance_does_not_count_inactive_cash_days_as_low_risk():
    dates, returns, states, active = fixture()
    result = active_budget_frame(dates, returns, states, 1, active, 5)
    row = result.iloc[5]
    a = np.std([-.02, .02], ddof=1)
    b = np.std([-.01, .01, -.01, .01], ddof=1)
    assert row.panic_active_days == 2 and row.learned_active_days == 4
    assert np.isclose(row.panic_budget, b/(a+b))
    original = budget_frame(dates, returns, states, 1, 5)
    assert row.panic_budget < original.panic_budget.iloc[5]


def test_one_active_observation_or_unknown_mask_retains_no_view_and_old_budget():
    dates, returns, states, active = fixture()
    active[3, 0] = 0
    result = active_budget_frame(dates, returns, states, 1, active, 5)
    assert result.risk_status.iloc[5] == "NO_VIEW_TOO_FEW_ACTIVE_DAYS_KEEP_BUDGET"
    assert result.panic_budget.iloc[5] == .5
    active[3, 0] = np.nan
    result = active_budget_frame(dates, returns, states, 1, active, 5)
    assert result.risk_status.iloc[5] == "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"


def test_future_activity_and_returns_cannot_change_past_and_full_account_calendar_stays():
    dates, returns, states, active = fixture()
    original = active_budget_frame(dates, returns, states, 1, active, 5)
    active[10:], returns[10:], states[10:] = 1, .9, [0., 1.]
    changed = active_budget_frame(dates, returns, states, 1, active, 5)
    assert_frame_equal(original.iloc[:10], changed.iloc[:10])
    assert len(changed) == len(dates) and changed.date.tolist() == list(dates)
