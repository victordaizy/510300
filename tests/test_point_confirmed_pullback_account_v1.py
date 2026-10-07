"""必要检查：阶段确认、先后信息、开盘风险空间、T+1及股息现金。"""
import numpy as np
import pandas as pd

from research.point_confirmed_pullback_account_v1 import signals, geometry, account
from research.point_account_nr7_inputs_v1 import PARENT_A


def phase_fixture():
    close = np.array([11., 10.5, 10., 10.7, 11.2, 12., 13., 12., 11., 10.7,
                      11.15, 11.2, 11.05, 11.3, 12., 13.2])
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)),
                         "open": close, "high": close + .05, "low": close - .05, "close": close,
                         "dividend": 0., "amount": [1000.] * 8 + [400., 400., 2000.] + [400.] * 5,
                         "volume": 100., "symbol": "510300.SH", "amount_unit": "CNY", "volume_unit": "share"})


def account_fixture():
    dates = pd.bdate_range("2020-01-01", periods=5)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.1,
                         "low": 9.9, "dividend": 0., "cash_shift": 0., "ac": 10.})
    parent = pd.DataFrame({"origin": dates, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": range(len(dates)), "es95": .02})
    sig = pd.DataFrame({"date": dates, "entry_event": [True, False, False, False, False],
                        "event_id": ["FIXED_FIRST_TURN"] + [None] * 4,
                        "stop_index": [9.] + [np.nan] * 4, "target_index": [13.] + [np.nan] * 4})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return data, parent, risks, sig, dividends


def test_confirmed_phase_is_prefix_invariant_and_first_turn_consumed_once():
    data = phase_fixture()
    full = signals(data)
    short = signals(data.iloc[:11].copy())
    pd.testing.assert_frame_equal(full.iloc[:11].reset_index(drop=True), short)
    assert full.entry_event.iloc[10]
    assert full.entry_event.sum() == 1
    assert full.event_status.iloc[13] == "FIRST_TURN_ALREADY_CONSUMED"
    assert full.high_index.iloc[10] + 2 <= 10


def test_contraction_must_be_known_before_turn_and_today_volume_does_not_replace_it():
    data = phase_fixture()
    data.loc[10, "amount"] = 1000000.
    assert signals(data).entry_event.iloc[10]
    data.loc[9, "amount"] = 2000.
    assert not signals(data).entry_event.iloc[10]
    assert signals(data).event_status.iloc[10] == "NO_PRIOR_AMOUNT_CONTRACTION"


def test_minimum_fee_and_slippage_reduce_planned_reward_risk():
    value = geometry(100, 10., 9.9, 10.2, "STRESS")
    assert value["planned_net_loss"] > 10.
    assert value["planned_net_reward"] < 20.
    assert value["net_reward_risk"] < 2.


def test_open_gap_outside_two_r_cancels_without_later_retry():
    data, parent, risks, sig, dividends = account_fixture()
    data.loc[1, "open"] = 10.5
    result = account(data, dividends, parent, risks, sig, "STRESS", "2020-01-02", "PULLBACK_ONLY")
    assert result["orders"].empty
    assert result["rejections"].reason.eq("OPEN_OUTSIDE_NET_TWO_R_GEOMETRY").any()
    assert result["daily"].shares.eq(0).all()


def test_first_fill_does_not_read_entry_day_close_and_exit_obeys_t_plus_one():
    data, parent, risks, sig, dividends = account_fixture()
    data.loc[1, ["close", "ac"]] = 8.95
    first = account(data, dividends, parent, risks, sig, "STRESS", "2020-01-02", "PULLBACK_ONLY")
    changed = data.copy()
    changed.loc[1, ["close", "ac", "high", "low"]] = [9.8, 9.8, 10.8, 8.8]
    second = account(changed, dividends, parent, risks, sig, "STRESS", "2020-01-02", "PULLBACK_ONLY")
    buy_a = first["orders"].loc[first["orders"].side.eq("BUY")].iloc[0]
    buy_b = second["orders"].loc[second["orders"].side.eq("BUY")].iloc[0]
    assert buy_a.quantity == buy_b.quantity and buy_a.fill_price == buy_b.fill_price
    trade = first["trades"].iloc[0]
    assert trade.entry_origin < trade.entry_date < trade.exit_date
    assert trade.exit_reason == "STRUCTURAL_STOP_CONFIRMED"
    np.testing.assert_allclose(first["daily"].accounting_error, 0., atol=1e-6)


def test_dividend_entitlement_receivable_payment_and_trade_cashflow_are_separate():
    data, parent, risks, sig, dividends = account_fixture()
    sig.loc[0, ["stop_index", "target_index"]] = [9.7, 11.]
    data.loc[1, ["close", "ac"]] = [10.1, 10.1]
    data.loc[2:, "dividend"] = [0.1, 0., 0.]
    data["cash_shift"] = data.dividend.cumsum()
    data.loc[2:, "open"] = [10., 11., 11.]
    data.loc[2:, "close"] = [10.95, 11., 11.]
    data["ac"] = data.close + data.cash_shift
    dividends = pd.DataFrame({"record_date": [data.date.iloc[1]], "ex_date": [data.date.iloc[2]],
                              "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [0.1]})
    result = account(data, dividends, parent, risks, sig, "STRESS", "2020-01-02", "PULLBACK_ONLY")
    daily = result["daily"]
    assert daily.receivable.iloc[1] > 0 and daily.dividend_paid.iloc[1] == 0
    assert daily.shares.iloc[2] == 0 and daily.receivable.iloc[2] > 0
    assert daily.receivable.iloc[3] == 0 and daily.dividend_paid.iloc[3] > 0
    trade = result["trades"].iloc[0]
    assert abs(trade.net_pnl - (trade.sell_net_cny + trade.dividend_cny - trade.buy_debit)) < 1e-8
    np.testing.assert_allclose(daily.equity, daily.cash + daily.shares * daily.close + daily.receivable, atol=1e-7)
