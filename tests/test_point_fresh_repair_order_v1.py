"""验证已知区间顺序、旧转强失效、未来隔离和实际下一开盘执行。"""
import numpy as np
import pandas as pd

from research import point_fresh_repair_order_inputs_v1 as rules
from research.point_fresh_repair_order_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A


def path():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=6),
                         "ac": [9.8, 9.8, 9.8, 9.8, 10.2, 9.8], "ema20": 10.,
                         "daily_hist": [-.1, -.1, -.1, .1, .1, -.1],
                         "up_volume_balance5": [-.2, -.2, .2, .2, .2, -.2],
                         "available": True, "atr20": .1})


def test_current_births_precede_price_confirmation():
    rows = rules.signals(path())
    r = rows.iloc[4]
    assert r.ordered_repair and r.known_under_start == 0
    assert r.current_volume_positive_start == 2 and r.current_macd_positive_start == 3
    assert not rows.iloc[:4].ordered_repair.any()


def test_old_positive_state_and_same_day_order_do_not_count_as_fresh_births():
    old = path()
    old.up_volume_balance5 = .2
    assert not rules.signals(old).ordered_repair.any()
    simultaneous = path()
    simultaneous.loc[2, "up_volume_balance5"] = -.2
    assert not rules.signals(simultaneous).ordered_repair.any()
    reverse = path()
    reverse.loc[1:, "daily_hist"] = .1
    assert not rules.signals(reverse).ordered_repair.any()


def test_future_suffix_cannot_change_past_sequence():
    full = rules.signals(path())
    for n in [3, 4, 5]:
        pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True), rules.signals(path().iloc[:n]), check_exact=True)


def test_missing_state_cannot_prove_a_positive_birth():
    missing = path()
    missing.loc[1, "up_volume_balance5"] = np.nan
    rows = rules.signals(missing)
    assert rows.status.iloc[1] == "NO_VIEW_PRICE_OR_VOLUME_OR_MACD"
    assert not rows.ordered_repair.any()


def test_entry_quantity_uses_prior_close_and_exit_is_next_legal_open():
    d = path()
    d["open"], d["high"], d["low"], d["close"] = 10., 10.3, 9.7, d.ac
    d["dividend"], d["cash_shift"] = 0., 0.
    seq = rules.signals(d)
    signals = rules.account_signals(d, seq, "FRESH_ORDERED_REPAIR")
    extra = d.iloc[-1:].copy()
    extra.date = d.date.iloc[-1]+pd.offsets.BDay()
    extended = pd.concat([d, extra], ignore_index=True)
    seq = rules.signals(extended)
    signals = rules.account_signals(extended, seq, "FRESH_ORDERED_REPAIR")
    div = pd.DataFrame({"record_date": pd.Series(dtype="datetime64[ns]"), "ex_date": pd.Series(dtype="datetime64[ns]"),
                        "payment_date": pd.Series(dtype="datetime64[ns]"), "cash_dividend_per_share": pd.Series(dtype=float)})
    parents = pd.DataFrame({"origin": extended.date, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": np.arange(len(extended)), "es95": .01})
    a = account(extended, div, parents, risks, signals, "STRESS", str(extended.date.iloc[1].date()), "ORDERED_ONLY")
    changed = extended.copy()
    changed.loc[5, ["ac", "close"]] = 10.5
    b = account(changed, div, parents, risks, signals, "STRESS", str(extended.date.iloc[1].date()), "ORDERED_ONLY")
    aa, bb = a["orders"].query("side=='BUY'").iloc[0], b["orders"].query("side=='BUY'").iloc[0]
    assert aa.quantity == bb.quantity and aa.fill_price == bb.fill_price
    trade = a["trades"].iloc[0]
    assert trade.entry_date == extended.date.iloc[5] and trade.exit_date == extended.date.iloc[6]
    assert trade.exit_reason == "REPAIR_PRICE_CONFIRMATION_FAILED_CLOSE"
    np.testing.assert_allclose(a["daily"].accounting_error, 0., atol=1e-7)
