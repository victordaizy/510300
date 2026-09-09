"""核对缺口、递推、边界、分红和通道信号的真实成交时钟。"""
import numpy as np
import pandas as pd
import pytest

from research.atr_trend_bands_inputs_v1 import band_frame
from research.event_clock_account_v1 import simulate_event_account


def prices(closes):
    closes = np.asarray(closes, float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(closes)), "open": closes,
        "high": closes, "low": closes, "close": closes, "previous_close": np.r_[np.nan, closes[:-1]],
        "dividend": 0., "wealth": closes / closes[0], "variance60": .0001})


def test_gap_true_range_and_wilder_seed_are_hand_calculable():
    data = prices([10, 12, 13, 12])
    result = band_frame(data, window=2, multiplier=1)
    np.testing.assert_allclose(result.true_range, [np.nan, .2, .1, .1], equal_nan=True, atol=1e-14)
    np.testing.assert_allclose(result.atr, [np.nan, np.nan, .15, .125], equal_nan=True, atol=1e-14)
    assert result.target.iloc[:2].isna().all() and result.target.iloc[2] == 0


def test_manual_crossing_path_has_expected_full_entry_and_exit():
    result = band_frame(prices([10, 10, 10, 12, 13, 11, 10, 12]), window=2, multiplier=1)
    np.testing.assert_allclose(result.target, [np.nan, np.nan, 0, 1, 1, 0, 0, 1], equal_nan=True)
    assert result.upper_band.iloc[3] == pytest.approx(1.)
    assert result.lower_band.iloc[5] == pytest.approx(1.2)


def test_touching_active_lower_band_does_not_exit():
    result = band_frame(prices([8, 8, 8, 12, 14, 12, 12, 10]), window=2, multiplier=1)
    assert result.wealth_close.iloc[5] == result.lower_band.iloc[5]
    assert result.target.iloc[5:7].eq(1).all() and result.target.iloc[7] == 0


def test_pure_cash_dividend_does_not_create_signal_price_gap():
    data = prices([10, 10, 9, 9, 9])
    data.loc[2, "dividend"] = 1
    data["wealth"] = 1.
    result = band_frame(data, window=2, multiplier=1)
    assert result.wealth_high.eq(1).all() and result.wealth_low.eq(1).all()
    assert result.true_range.iloc[1:].eq(0).all() and result.target.iloc[2:].eq(0).all()


def test_missing_input_resets_warmup_and_preserves_no_view():
    data = prices([10] * 9)
    data.loc[4, "high"] = np.nan
    result = band_frame(data, window=2, multiplier=1)
    assert result.target.iloc[4:6].isna().all()
    assert result.source_state.iloc[4] == "NO_VIEW_INPUT_RESET"
    assert result.source_state.iloc[6] == "VIEW_INITIAL_DOWN_STATE"


@pytest.mark.parametrize("column,value", [("high", np.inf), ("low", -1.), ("dividend", -1.)])
def test_invalid_prices_fail_instead_of_repairing(column, value):
    data = prices([10, 11, 12])
    data.loc[1, column] = value
    with pytest.raises(ValueError):
        band_frame(data, window=2)


def test_future_prices_cannot_change_prior_bands():
    data = prices([10, 10, 10, 12, 13, 11, 10, 12])
    before = band_frame(data, window=2, multiplier=1).iloc[:5]
    changed = data.copy()
    changed.loc[5:, ["high", "low", "close", "wealth"]] *= 2
    pd.testing.assert_frame_equal(before, band_frame(changed, window=2, multiplier=1).iloc[:5])


def test_wealth_scale_changes_do_not_change_targets():
    data = prices([10, 10, 10, 12, 13, 11, 10, 12])
    expected = band_frame(data, window=2, multiplier=1).target
    data["wealth"] *= 100
    np.testing.assert_allclose(band_frame(data, window=2, multiplier=1).target, expected, equal_nan=True)


def test_next_open_execution_and_actual_costed_round_trips():
    data = prices([10, 10, 10, 10.5, 10.8, 10.2, 10.1, 10.4, 10.4, 10.4])
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    targets = band_frame(data, window=2, multiplier=1).target.to_numpy()
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(data.date.iloc[1].date()), "通道测试", targets=targets, event_mask=np.ones(len(data), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[4], data.date.iloc[8]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[6], data.date.iloc[9]]
    assert decisions.iloc[:2].reference_weight.isna().all()
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.iloc[-1].shares == 0
