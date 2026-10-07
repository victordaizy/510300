"""验证固定分组、真实未知边界与信息滞后，避免扩散观察读入未来。"""
import numpy as np
import pandas as pd

from research.broker_fixed_cohort_inputs_v1 import Panel, make_cohort, observe


def panel():
    dates = pd.bdate_range("2020-01-01", periods=55)
    symbols = pd.Index([f"S{i:03d}" for i in range(300)])
    logs = np.full((55, 300), .001)
    returns = np.tile(np.linspace(.02, -.02, 300), (55, 1))
    return Panel(dates, symbols, logs, np.ones((55, 300), bool), np.ones(55, bool), returns)


def test_fixed_groups_use_prior_close_and_remain_unchanged():
    p = panel()
    cohort = make_cohort(p, 25)
    assert cohort.leaders == tuple(range(150)) and cohort.followers == tuple(range(150, 300))
    p.return20[25:] = p.return20[25:, ::-1]
    assert make_cohort(p, 25) == cohort
    assert observe(p, cohort, 30)["leaders_size"] == 150


def test_observation_has_one_complete_session_lag():
    p = panel()
    cohort = make_cohort(p, 25)
    before = observe(p, cohort, 30)
    p.logs[30:] = -.5
    assert observe(p, cohort, 30) == before
    assert before["observation_source_date"] < before["date"]


def test_missing_post_anchor_return_remains_unknown_with_bounds():
    p = panel()
    cohort = make_cohort(p, 25)
    p.logs[27, :4] = np.nan
    row = observe(p, cohort, 30)
    assert row["leaders_known"] == 146 and row["leaders_unknown"] == 4
    assert np.isclose(row["leaders_positive_lower"], 146 / 150)
    assert row["leaders_positive_upper"] == 1 and not row["leaders_view_allowed"]


def test_no_future_membership_replaces_fixed_cohort():
    p = panel()
    cohort = make_cohort(p, 25)
    p.members[27:, :10] = False
    row = observe(p, cohort, 30)
    assert cohort.leaders == tuple(range(150)) and row["leaders_size"] == 150
    assert row["leaders_still_members"] == 140 and not row["view_allowed"]


def test_ties_have_deterministic_order_and_inadequate_anchor_abstains():
    p = panel()
    p.return20[:] = 0
    assert make_cohort(p, 25).leaders == tuple(range(150))
    p.return20[24, :7] = np.nan
    cohort = make_cohort(p, 25)
    assert cohort.status == "NO_VIEW_ANCHOR_MEMBERS_OR_RETURNS" and cohort.eligible_count == 293
    assert not observe(p, cohort, 30)["view_allowed"]
