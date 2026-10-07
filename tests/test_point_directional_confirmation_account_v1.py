"""必要检查：确认时钟、股息、下一开盘、T+1及完整资金账户。"""
import numpy as np
import pandas as pd
import pytest

from research.point_directional_confirmation_account_v1 import signals, account
from research.point_account_nr7_inputs_v1 import PARENT_A


def market(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)),
                         "open": close, "close": close, "high": close + .01,
                         "low": close - .01, "dividend": 0., "cash_shift": 0., "ac": close})


def fixture():
    data = market([10.] * 5)
    parent = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": range(len(data)), "es95": .02})
    events = pd.DataFrame({"date": data.date, "entry_event": [True, False, False, False, False],
                           "direction_state": [1, 1, -1, -1, -1],
                           "event_id": ["UP_KNOWN_AT_CLOSE"] + [None] * 4,
                           "stop_index": np.nan, "target_index": np.nan})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return data, parent, risks, events, dividends


def test_event_is_at_confirmation_and_prefix_does_not_change():
    data = market([100., 90., 94., 95., 98., 104., 99., 98., 97.])
    result = signals(data)
    assert result.index[result.entry_event].tolist() == [3]
    assert result.index[result.exit_event].tolist() == [1, 7]
    assert result.confirmed_extreme_index.iloc[3] == 1
    assert result.confirmed_extreme_index.iloc[7] == 5
    assert result.confirmation_index.iloc[3] == 3
    for end in [3, 4, 7, 8]:
        pd.testing.assert_frame_equal(result.iloc[:end].reset_index(drop=True), signals(data.iloc[:end]))


def test_ex_dividend_price_drop_is_not_economic_direction_change():
    data = market([10., 10.6, 10., 10., 10.])
    data.loc[2, "dividend"] = .6
    result = signals(data)
    assert result.entry_event.iloc[1]
    assert not result.exit_event.any()
    assert result.direction_state.iloc[-1] == 1
    np.testing.assert_allclose(result.wealth_index.iloc[1:], 1.06)


def test_missing_price_is_rejected_instead_of_filled_or_skipped():
    data = market([10., 10.1, 10.2])
    data.loc[1, "close"] = np.nan
    with pytest.raises(ValueError, match="缺失"):
        signals(data)


def test_entry_day_close_cannot_change_first_open_fill_and_exit_is_t_plus_one():
    data, parent, risks, events, dividends = fixture()
    events.loc[1, "direction_state"] = -1
    first = account(data, dividends, parent, risks, events, "STRESS", "2020-01-02", "DIRECTIONAL_ONLY")
    changed = data.copy()
    changed.loc[1, ["close", "ac", "high", "low"]] = [9.8, 9.8, 10.5, 9.5]
    second = account(changed, dividends, parent, risks, events, "STRESS", "2020-01-02", "DIRECTIONAL_ONLY")
    buy_a = first["orders"].query("side=='BUY'").iloc[0]
    buy_b = second["orders"].query("side=='BUY'").iloc[0]
    assert buy_a.quantity == buy_b.quantity and buy_a.fill_price == buy_b.fill_price
    trade = first["trades"].iloc[0]
    assert trade.entry_origin < trade.entry_date < trade.exit_date
    assert trade.exit_reason == "DIRECTIONAL_DOWN_CONFIRMATION"
    np.testing.assert_allclose(first["daily"].accounting_error, 0., atol=1e-6)


def test_component_enters_only_when_parent_explicit_zero_and_ignores_unknown():
    data, parent, risks, events, dividends = fixture()
    parent.loc[0, PARENT_A] = np.nan
    result = account(data, dividends, parent, risks, events, "BASE", "2020-01-02", "A_PLUS_DIRECTIONAL")
    assert result["orders"].empty
    assert result["daily"].shares.eq(0).all()
    parent.loc[0, PARENT_A] = 0.
    result = account(data, dividends, parent, risks, events, "BASE", "2020-01-02", "A_PLUS_DIRECTIONAL")
    assert result["orders"].side.eq("BUY").sum() == 1
    assert result["trades"].source.eq("DIRECTIONAL").all()


def test_dividend_receivable_survives_sale_and_trade_cashflow_is_exact():
    data, parent, risks, events, dividends = fixture()
    data.loc[1, ["close", "ac"]] = [10.1, 10.1]
    data.loc[2, "dividend"] = .1
    data["cash_shift"] = data.dividend.cumsum()
    data.loc[2:, "open"] = [10., 11., 11.]
    data.loc[2:, "close"] = [10.95, 11., 11.]
    data["ac"] = data.close + data.cash_shift
    dividends = pd.DataFrame({"record_date": [data.date.iloc[1]], "ex_date": [data.date.iloc[2]],
                              "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    result = account(data, dividends, parent, risks, events, "STRESS", "2020-01-02", "DIRECTIONAL_ONLY")
    daily, trade = result["daily"], result["trades"].iloc[0]
    assert daily.receivable.iloc[1] > 0 and daily.dividend_paid.iloc[1] == 0
    assert daily.shares.iloc[2] == 0 and daily.receivable.iloc[2] > 0
    assert daily.receivable.iloc[3] == 0 and daily.dividend_paid.iloc[3] > 0
    assert abs(trade.net_pnl - (trade.sell_net_cny + trade.dividend_cny - trade.buy_debit)) < 1e-8
    np.testing.assert_allclose(daily.equity, daily.cash + daily.shares * daily.close + daily.receivable, atol=1e-7)
