"""检验手算路径、初始化、相等边界、分红、缺失和真实下一开盘成交。"""
import numpy as np
import pandas as pd
import pytest

from research.parabolic_reversal_inputs_v1 import parabolic_frame
from research.event_clock_account_v1 import simulate_event_account


def bars(high, low):
    high, low = np.asarray(high, float), np.asarray(low, float)
    close = (high + low) / 2
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)), "open": close, "high": high, "low": low,
        "close": close, "previous_close": np.r_[np.nan, close[:-1]], "dividend": 0., "wealth": close, "variance60": .0001})


def test_manual_up_and_down_paths_match_sar_and_acceleration():
    up = parabolic_frame(bars([10, 11, 12, 13], [9, 10, 11, 12]))
    down = parabolic_frame(bars([11, 10, 9, 8], [10, 9, 8, 7]))
    np.testing.assert_allclose(up.sar_level, [np.nan, 9., 9.04, 9.1584], atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(down.sar_level, [np.nan, 11., 10.96, 10.8416], atol=1e-13, equal_nan=True)
    assert up.target.iloc[1:].eq(1).all() and down.target.iloc[1:].eq(0).all()
    np.testing.assert_allclose(up.acceleration.iloc[1:], [.02, .04, .06])


def test_equal_directional_moves_start_up_and_no_new_extreme_keeps_acceleration():
    frame = parabolic_frame(bars([10, 11, 11, 11], [9, 10, 10, 10]))
    assert frame.target.iloc[1:].eq(1).all() and frame.acceleration.iloc[1:].eq(.02).all()
    tied = parabolic_frame(bars([10, 10.5], [9, 8.5]))
    assert tied.initialized.iloc[1] == 1 and tied.reversed.iloc[1] == 1 and tied.target.iloc[1] == 0


def test_equal_sar_touch_reverses_and_resets_speed():
    data = bars([10, 11, 12, 13], [9, 10, 11, 12])
    touch = parabolic_frame(data).next_sar.iloc[2]
    data.loc[3, ["low", "close", "wealth"]] = touch
    frame = parabolic_frame(data)
    assert frame.sar_input.iloc[3] == touch and frame.target.iloc[3] == 0
    assert frame.acceleration.iloc[3] == .02 and frame.sar_level.iloc[3] == 13.


def test_acceleration_is_capped_and_next_sar_respects_recent_range():
    data = bars(np.arange(10, 40), np.arange(9, 39))
    frame = parabolic_frame(data)
    assert frame.acceleration.max() == .2
    assert (frame.next_sar.iloc[2:] <= np.minimum(frame.wealth_low.iloc[2:], frame.wealth_low.shift().iloc[2:])).all()


def test_pure_cash_distribution_preserves_indicator_path():
    base = bars([10.5, 10.7, 10.9, 11.1, 11.3, 11.5], [9.5, 9.7, 9.9, 10.1, 10.3, 10.5])
    distributed = base.copy()
    distributed.loc[2:, ["open", "high", "low", "close"]] -= 1
    distributed.loc[2, "dividend"] = 1
    scale_after_distribution = (base.close.iloc[2] - 1) / base.close.iloc[2]
    distributed.loc[3:, ["open", "high", "low", "close"]] = base.loc[3:, ["open", "high", "low", "close"]].to_numpy() * scale_after_distribution
    pd.testing.assert_frame_equal(parabolic_frame(base), parabolic_frame(distributed))


def test_missing_input_requires_two_new_complete_bars():
    data = bars(np.arange(10, 18), np.arange(9, 17))
    data.loc[3, "high"] = np.nan
    frame = parabolic_frame(data)
    assert frame.target.iloc[3:5].isna().all() and frame.initialized.iloc[5] == 1
    assert frame.source_state.iloc[3] == "NO_VIEW_INPUT_RESET"


@pytest.mark.parametrize("column,value", [("high", np.inf), ("low", -1.), ("dividend", -1.)])
def test_invalid_input_is_rejected(column, value):
    data = bars([10, 11, 12], [9, 10, 11])
    data.loc[1, column] = value
    with pytest.raises(ValueError):
        parabolic_frame(data)


def test_future_data_does_not_change_prefix():
    data = bars([10, 11, 12, 13, 12, 11, 12], [9, 10, 11, 12, 11, 10, 11])
    old = parabolic_frame(data).iloc[:4]
    data.loc[4:, ["high", "low", "close", "wealth"]] *= 2
    pd.testing.assert_frame_equal(old, parabolic_frame(data).iloc[:4])


def test_next_open_execution_and_blocked_request_uses_latest_complete_state():
    data = bars([10.2, 10.4, 10.6, 10.8, 10.6, 10.7, 10.8, 10.8], [9.8, 10., 10.2, 10.4, 9.5, 9.6, 9.7, 9.7])
    frame = parabolic_frame(data)
    assert frame.target.iloc[1:4].eq(1).all() and frame.target.iloc[4:6].eq(0).all()
    data.loc[5, "open"] = (data.close.iloc[4] - data.dividend.iloc[5]) * .9
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(data.date.iloc[1].date()), "转向测试",
        targets=frame.target.to_numpy(), event_mask=np.ones(len(data), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == data.date.iloc[2]
    assert ledger.loc[ledger.date.eq(data.date.iloc[5]), "status"].iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == data.date.iloc[6]
    assert decisions.reference_weight.iloc[0] != decisions.reference_weight.iloc[0]
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0
