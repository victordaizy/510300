"""验证实际进入锚点、三行业信息、公布钟、未知分支和成交时序。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research import broker_stage_policy_inputs_v1 as original
from research import industry_leader_failure_inputs_v1 as candidate
from research.broker_fixed_cohort_inputs_v1 import Panel


def setup(policy=candidate.POLICIES[0]):
    dates = pd.bdate_range("2020-01-01", periods=60)
    symbols = pd.Index([f"S{i:03d}" for i in range(300)])
    logs = np.full((60, 300), .001)
    logs[25:30, :150] = -.002
    returns20 = np.tile(np.repeat(np.array([.06, .04, .02, 0., -.02, -.04]), 50), (60, 1))
    panel = Panel(dates, symbols, logs, np.ones((60, 300), bool), np.ones(60, bool), returns20)
    rows = []
    for date in dates:
        rows.extend({"date": date, "symbol": symbol, "industry_key": f"G{i//50}", "major_name": f"行业{i//50}",
            "row_source_known": True} for i, symbol in enumerate(symbols))
    calendar = pd.DataFrame({"date": dates, "membership_source_date": [pd.NaT, *dates[:-1]],
        "available_at": [pd.NaT, *(date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59) for date in dates[:-1])],
        "row_source_snapshot_eligible": True, "snapshot_id": "SYNTHETIC", "snapshot_taxonomy": "TEST", "source_age_calendar_days": 10.})
    source = candidate.IndustryData(panel, pd.DataFrame(rows), calendar)
    data = pd.DataFrame({"date": dates, "close": 4 * np.exp(.001 * np.arange(60)), "dividend": 0.})
    active = {"entry_idx": 25, "entry_date": dates[25], "cycle_id": 1}
    controller = candidate.FailureController(source, data, policy, lambda *_: None)
    return source, data, active, controller


def call(controller, active, data, i=30):
    return controller(active, SimpleNamespace(date=data.date.iloc[i]), i, "STAGE_ENTRY_AND_EXIT")


def test_real_entry_anchor_lagged_five_and_next_open():
    _, data, active, controller = setup()
    assert call(controller, active, data) == f"{candidate.POLICIES[0]}_FIVE_SOURCE_CLOSES"
    row = controller.checks[0]
    assert row["anchor_source_date"] == data.date.iloc[24]
    assert row["observation_source_date"] == data.date.iloc[29]
    assert row["execution_date"] == data.date.iloc[31]
    assert row["fixed_leader_ids"] == "G0;G1;G2"
    assert row["leader_mean_cumulative_return"] < 0 < row["etf_lagged_five_return"]


def test_only_fifth_day_unknown_does_not_retry():
    source, data, active, controller = setup()
    source.panel.allowed[24] = False
    assert call(controller, active, data, 29) is None
    assert call(controller, active, data, 30) is None
    assert call(controller, active, data, 31) is None
    assert len(controller.checks) == 1


def test_price_controls_ignore_industry_direction():
    for policy in candidate.POLICIES:
        source, data, active, controller = setup(policy)
        source.panel.logs[25:30, :150] = .001
        assert (call(controller, active, data) is None) == (policy == candidate.POLICIES[0])


def test_missing_leading_group_keeps_main_and_same_coverage_control():
    for policy in candidate.POLICIES:
        source, data, active, controller = setup(policy)
        source.panel.logs[27, :2] = np.nan
        assert (call(controller, active, data) is not None) == (policy == candidate.POLICIES[2])
        assert not controller.checks[0]["leader_view_allowed"]


def test_missing_followers_is_reported_but_not_leader_only_condition():
    source, data, active, controller = setup()
    source.panel.logs[27, 150:152] = np.nan
    assert call(controller, active, data) is not None
    assert controller.checks[0]["leader_view_allowed"]
    assert not controller.checks[0]["full_description_view_allowed"]


def test_classification_published_after_entry_open_is_unknown():
    source, data, active, _ = setup()
    source.slots[data.date.iloc[25]]["available_at"] = data.date.iloc[25].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=12)
    controller = candidate.FailureController(source, data, candidate.POLICIES[0], lambda *_: None)
    assert call(controller, active, data) is None
    assert not controller.checks[0]["leader_view_allowed"]


def test_same_day_future_and_later_classification_do_not_change_decision():
    source, data, active, controller = setup()
    reason = call(controller, active, data)
    before = controller.checks[0].copy()
    source.panel.logs[30:] = 10.
    source.panel.return20[25:] = -source.panel.return20[25:]
    source.members.loc[source.members.date.gt(data.date.iloc[25]), "industry_key"] = "FUTURE_CLASS"
    data.loc[30:, "close"] = 1000.
    data.loc[30:, "dividend"] = 999.
    after_controller = candidate.FailureController(source, data, candidate.POLICIES[0], lambda *_: None)
    assert call(after_controller, active, data) == reason
    assert after_controller.checks[0] == before


def test_original_exit_priority_and_private_engine_isolation():
    _, data, active, controller = setup()
    controller.original_exit = lambda *_: "KNOWN_STRUCTURAL_EXIT"
    assert call(controller, active, data) == "KNOWN_STRUCTURAL_EXIT"
    assert not controller.checks[0]["extra_exit"]
    frozen_function = original.exit_decision
    private = candidate.private_engine()
    private.exit_decision = controller
    assert original.exit_decision is frozen_function
    assert candidate.private_engine().exit_decision is not controller


def test_dividend_can_supply_positive_etf_condition():
    source, data, active, _ = setup()
    data["close"] = 4.
    no_dividend = candidate.FailureController(source, data, candidate.POLICIES[0], lambda *_: None)
    assert call(no_dividend, active, data) is None
    data.loc[27, "dividend"] = .02
    dividend = candidate.FailureController(source, data, candidate.POLICIES[0], lambda *_: None)
    assert call(dividend, active, data) is not None
    assert np.isclose(dividend.checks[0]["etf_lagged_five_return"], .005)
