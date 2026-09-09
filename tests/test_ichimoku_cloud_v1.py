"""云图区间中点、历史位移、分红中性与三值信号真实账户。"""
import numpy as np
import pandas as pd
from research.ichimoku_cloud_inputs_v1 import cloud_frame
from research.known_price_signal_account_v1 import simulate_known_price_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def market(n=210):
    dates = pd.bdate_range("2018-01-01", periods=n)
    price = 10+np.arange(n)*.025+np.sin(np.arange(n)/6)*.2
    return pd.DataFrame({"date": dates, "open": price, "close": price, "high": price+.1, "low": price-.1,
                         "previous_close": np.r_[np.nan, price[:-1]], "dividend": 0., "wealth": price/price[0]})


def rules(args):
    args[5]["entry"] = args[5]["entry"].astype(float)
    args[5]["exit"][1] = args[5]["exit"][1].astype(float)
    return args


def test_extreme_midpoints_and_full_twenty_six_day_cloud_delay_are_exact():
    data = market()
    data.loc[75, "high"] += 5.
    out = cloud_frame(data)
    assert out.entry_signal.iloc[:78].isna().all() and out.entry_signal.iloc[78:].notna().all()
    assert out.wealth_high.isna().iloc[0]
    for t in [78, 91, 145, 209]:
        for name, window in [("conversion", 9), ("base", 26), ("computed_span_b", 52)]:
            high = out.wealth_high.iloc[t-window+1:t+1].max()
            low = out.wealth_low.iloc[t-window+1:t+1].min()
            assert out[name].iloc[t] == (high+low)/2
        assert out.current_cloud_a.iloc[t] == out.computed_span_a.iloc[t-26]
        assert out.current_cloud_b.iloc[t] == out.computed_span_b.iloc[t-26]
        assert out.cloud_calculation_date.iloc[t] == data.date.iloc[t-26]
    assert abs(out.conversion.iloc[78]-out.wealth_close.iloc[70:79].mean()) > .1


def test_future_price_changes_cannot_change_a_previous_cloud_or_decision():
    data = market(); before = cloud_frame(data)
    cut = 120
    altered = data.copy()
    altered.loc[cut+1:, ["high", "low", "open", "close"]] *= 1.7
    altered["previous_close"] = altered.close.shift(1)
    altered["wealth"] = altered.close/altered.close.iloc[0]
    after = cloud_frame(altered)
    pd.testing.assert_frame_equal(before.iloc[:cut+1], after.iloc[:cut+1])
    assert not before.iloc[cut+1:].equals(after.iloc[cut+1:])


def test_dividend_price_drop_does_not_create_a_false_economic_breakdown():
    data = market(160)
    data[["open", "high", "low", "close"]] = 10.
    data.loc[90:, ["open", "high", "low", "close"]] = 9.5
    data["previous_close"] = data.close.shift(1)
    data.loc[90, "dividend"] = .5
    data["wealth"] = 1.
    out = cloud_frame(data)
    np.testing.assert_allclose(out.wealth_close.iloc[1:], 1., atol=1e-12, rtol=0)
    assert out.entry_signal.dropna().eq(0).all() and out.exit_signal.dropna().eq(0).all()


def test_unknown_entry_is_not_false_but_a_known_cloud_exit_remains_actionable():
    data = market()
    data.loc[90, "high"] = np.nan
    data.loc[90, ["close", "low"]] *= .6
    out = cloud_frame(data)
    assert np.isnan(out.entry_signal.iloc[90])
    assert out.exit_signal.iloc[90] == 1. and out.exit_by_cloud.iloc[90]
    assert np.isnan(out.entry_signal.iloc[167]) and np.isfinite(out.entry_signal.iloc[168])


def test_fully_known_signals_reproduce_original_cash_dividend_and_execution_account():
    args = rules(fixture())
    args[5]["exit"][1][2] = 1.
    args[5]["entry"][5] = 0.
    old_ledger, old_decisions, old_cycles = simulate_rearmed_exit(*args)
    ledger, decisions, cycles = simulate_known_price_exit(*args)
    pd.testing.assert_frame_equal(ledger, old_ledger)
    pd.testing.assert_frame_equal(cycles, old_cycles)
    pd.testing.assert_frame_equal(decisions[old_decisions.columns], old_decisions)


def test_unknown_flat_condition_never_rearms_and_unknown_after_unfilled_buy_never_retries():
    args = rules(fixture())
    args[5]["exit"][1][2] = 1.
    args[5]["entry"][[3, 4]] = np.nan
    args[5]["entry"][6] = 0.
    ledger, decisions, cycles = simulate_known_price_exit(*args)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[8]]
    assert decisions.loc[decisions.origin_index.isin([3, 4, 5]), "entry_rearmed"].eq(False).all()
    assert decisions.loc[decisions.origin_index.isin([3, 4]), "reference_weight"].isna().all()
    args = rules(fixture())
    args[0].loc[1, "open"] = 11.
    args[5]["entry"][1] = np.nan
    ledger, decisions, cycles = simulate_known_price_exit(*args)
    assert ledger.status.iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.requested_quantity.iloc[1] == 0
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == args[0].date.iloc[3]


def test_locked_sale_survives_unknown_signals_and_remains_next_open_t_plus_one():
    args = rules(fixture())
    args[5]["exit"][1][1] = 1.
    args[0].loc[2, "open"] = 9.
    args[5]["entry"][2] = np.nan
    args[5]["exit"][1][2] = np.nan
    ledger, decisions, cycles = simulate_known_price_exit(*args)
    assert ledger.status.iloc[1] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert decisions.loc[decisions.origin_index.eq(2), "requested_quantity"].iloc[0] < 0
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[3]]
    assert cycles.holding_intervals.ge(1).all() and ledger.shares.iloc[-1] == 0
    assert ledger.accounting_error.abs().max() < 1e-7
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0


def test_simultaneous_entry_and_exit_has_exit_priority_without_consuming_entry_right():
    args = rules(fixture())
    args[5]["exit"][1][0:2] = 1.
    ledger, decisions, cycles = simulate_known_price_exit(*args)
    assert ledger.filled_quantity.iloc[:2].eq(0).all()
    assert decisions.entry_rearmed.iloc[:2].eq(True).all()
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == args[0].date.iloc[3]
