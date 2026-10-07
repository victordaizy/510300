"""验证真实进入锚点、单次失效、未知对照及原引擎隔离。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research import broker_cohort_failure_inputs_v1 as candidate
from research import broker_stage_policy_inputs_v1 as original
from research.broker_fixed_cohort_inputs_v1 import Panel


def setup(policy=candidate.POLICIES[0]):
    dates = pd.bdate_range("2020-01-01", periods=60)
    logs = np.full((60, 300), -.001)
    returns = np.tile(np.linspace(.02, -.02, 300), (60, 1))
    panel = Panel(dates, pd.Index([f"S{i:03d}" for i in range(300)]), logs,
        np.ones((60, 300), bool), np.ones(60, bool), returns)
    data = pd.DataFrame({"date": dates, "close": 4 * np.exp(-.001 * np.arange(60)), "dividend": 0.})
    controller = candidate.FailureController(panel, data, policy, lambda *_: None)
    active = {"entry_idx": 25, "entry_date": dates[25], "cycle_id": 1}
    return panel, data, controller, active


def call(controller, active, data, i):
    return controller(active, SimpleNamespace(date=data.date.iloc[i]), i, "STAGE_ENTRY_AND_EXIT")


def test_joint_failure_at_fifth_source_close():
    _, data, controller, active = setup()
    reason = call(controller, active, data, 30)
    assert reason == "COHORT_AND_PRICE_FAILURE_FIVE_SOURCE_CLOSES_FAILED"
    check = controller.checks[0]
    assert check["cohort_source_date"] == data.date.iloc[24]
    assert check["observation_source_date"] == data.date.iloc[29]
    assert check["execution_date"] == data.date.iloc[31]


def test_only_one_fixed_day_is_checked():
    _, data, controller, active = setup()
    assert call(controller, active, data, 29) is None
    assert call(controller, active, data, 31) is None
    assert not controller.checks


def test_price_controls_do_not_use_group_direction():
    for policy in candidate.POLICIES:
        panel, data, controller, active = setup(policy)
        panel.logs[25:30, :150] = .001
        reason = call(controller, active, data, 30)
        assert (reason is None) == (policy == candidate.POLICIES[0])


def test_unknown_source_preserves_main_and_common_control_only():
    for policy in candidate.POLICIES:
        panel, data, controller, active = setup(policy)
        panel.allowed[24] = False
        reason = call(controller, active, data, 30)
        assert (reason is not None) == (policy == candidate.POLICIES[2])


def test_same_day_and_future_returns_cannot_change_failure():
    panel, data, controller, active = setup()
    reason = call(controller, active, data, 30)
    panel.logs[30:] = 10
    panel.return20[25:] = -panel.return20[25:]
    assert call(controller, active, data, 30) == reason


def test_original_exit_has_priority():
    _, data, controller, active = setup()
    controller.original_exit = lambda *_: "KNOWN_STRUCTURAL_LOW_FAILED"
    assert call(controller, active, data, 30) == "KNOWN_STRUCTURAL_LOW_FAILED"
    assert not controller.checks[0]["extra_exit"]


def test_private_module_does_not_modify_frozen_global_engine():
    function = original.exit_decision
    private = candidate.private_engine()
    private.exit_decision = lambda *_: "PRIVATE_ONLY"
    assert original.exit_decision is function
    assert candidate.private_engine().exit_decision is not private.exit_decision


def test_etf_dividend_is_included_in_same_interval():
    panel, data, _, active = setup()
    data["close"] = 4.
    data.loc[27, "dividend"] = .02
    controller = candidate.FailureController(panel, data, candidate.POLICIES[0], lambda *_: None)
    assert call(controller, active, data, 30) is None
    assert np.isclose(controller.checks[0]["etf_lagged_five_return"], .005)
