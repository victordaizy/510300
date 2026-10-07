"""验证观察末端保留真实持仓、执行原退出，以及未来扩展不改变已知前段。"""
import numpy as np
import pandas as pd
import pytest

from research import reference_observation_accounts_v1 as observation


def inputs():
    dates = pd.bdate_range("2024-03-04", periods=8)
    frame = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .0001})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    config = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0004, "minimum": 5., "slippage": .001}
    return frame, dividends, config, cost


def execute(kind, frame, dividends, config, cost, exit_last=False, next_day=None):
    entry = np.ones(len(frame), dtype=int)
    exit_flag = np.zeros(len(frame), dtype=bool)
    targets = np.ones(len(frame)) * .5
    if exit_last:
        exit_flag[-2] = True
        targets[-2:] = 0.
    rule = {"entry": entry, "exit": {1: exit_flag}}
    spec = {"cooldown": 2, "modes": {1: {"loss": .06, "trail": .08, "take": None, "days": 60}}}
    next_day = next_day if next_day is not None else frame.date.iloc[-1] + pd.offsets.BDay(1)
    args = (frame, dividends, config, cost, str(frame.date.iloc[1].date()))
    if kind == "event":
        return observation.observe_event_account(*args, "TEST_TARGET", targets=targets, event_mask=np.ones(len(frame), bool), next_execution_date=next_day)
    if kind == "rearmed":
        return observation.observe_rearmed_exit(*args, rule, spec, next_execution_date=next_day)
    return observation.observe_price_policy(*args, rule, spec, next_execution_date=next_day)


@pytest.mark.parametrize("kind", ["rearmed", "event", "price"])
def test_last_complete_close_does_not_force_exit(kind):
    frame, dividends, config, cost = inputs()
    result = execute(kind, frame, dividends, config, cost)
    ledger, decisions = result[:2]
    assert ledger.shares.iloc[-1] > 0
    assert ledger.mark_clock.eq("CLOSE").all()
    assert decisions.origin.iloc[-1] == frame.date.iloc[-1]
    assert decisions.execution_date.iloc[-1] > frame.date.iloc[-1]
    if len(result) == 3:
        assert pd.isna(result[2].exit_date.iloc[-1])
        assert "观察截止" in result[2].exit_reasons.iloc[-1]


@pytest.mark.parametrize("kind", ["rearmed", "event", "price"])
def test_real_exit_request_still_executes_on_last_day(kind):
    frame, dividends, config, cost = inputs()
    ledger, decisions = execute(kind, frame, dividends, config, cost, exit_last=True)[:2]
    assert ledger.filled_quantity.iloc[-1] < 0
    assert ledger.shares.iloc[-1] == 0
    assert ledger.mark_clock.iloc[-1] == "CLOSE"
    assert decisions.origin.iloc[-1] == frame.date.iloc[-1]


@pytest.mark.parametrize("kind", ["rearmed", "event", "price"])
def test_observed_prefix_unchanged_after_new_bars(kind):
    frame, dividends, config, cost = inputs()
    cutoff = frame.date.iloc[5]
    short = execute(kind, frame.iloc[:6].copy(), dividends, config, cost, next_day=frame.date.iloc[6])
    frame.loc[6:, ["open", "close"]] = 10.5
    frame.loc[7, "previous_close"] = 10.5
    complete = execute(kind, frame, dividends, config, cost)
    pd.testing.assert_frame_equal(short[0].reset_index(drop=True), complete[0].loc[complete[0].date.le(cutoff)].reset_index(drop=True))
    pd.testing.assert_frame_equal(short[1].reset_index(drop=True), complete[1].loc[complete[1].origin.le(cutoff)].reset_index(drop=True))


def test_no_unknown_price_row_or_backdated_execution_day():
    frame, _, _, _ = inputs()
    with pytest.raises(ValueError, match="晚于"):
        observation.validate_observation_boundary(frame, frame.date.iloc[-1])
    frame.loc[len(frame)-1, "open"] = np.nan
    with pytest.raises(ValueError, match="未来价格"):
        observation.validate_observation_boundary(frame, frame.date.iloc[-1] + pd.offsets.BDay(1))
