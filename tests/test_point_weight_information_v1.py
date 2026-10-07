"""仓位强弱诊断只验证必要的调整、股息和时序边界。"""
import numpy as np
import pandas as pd

from research.point_weight_information_inputs_v1 import weight_account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def inputs():
    dates = pd.bdate_range("2024-01-02", periods=8)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "dividend": 0.})
    parents = pd.DataFrame({"origin": dates, PARENT_A: [.1, .4, .35, .35, 0., 0., 0., 0.]})
    risks = pd.DataFrame({"idx": np.arange(8), "es95": .01})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return data, parents, risks, dividends


def test_saved_weights_adjust_shares_but_do_not_manufacture_new_cycles():
    d, p, r, div = inputs()
    a = weight_account(d, div, p, r, "STRESS", d.date.iloc[1])
    assert a["orders"].side.tolist() == ["BUY", "BUY", "SELL"]
    assert a["orders"].date.tolist() == [d.date.iloc[1], d.date.iloc[2], d.date.iloc[5]]
    assert len(a["trades"]) == 1
    assert a["trades"].status.iloc[0] == "COMPLETE"
    verify_account(a)


def test_prefix_is_exact_and_no_terminal_forced_exit():
    d, p, r, div = inputs()
    full = weight_account(d, div, p, r, "STRESS", d.date.iloc[1])
    short = weight_account(d.iloc[:4], div, p, r, "STRESS", d.date.iloc[1])
    pd.testing.assert_frame_equal(full["daily"].iloc[:3], short["daily"])
    assert short["terminal"]["open_shares"] > 0
    assert short["trades"].status.iloc[0] == "RIGHT_CENSORED"


def test_dividend_belongs_to_record_quantity_before_subsequent_resize():
    d, p, r, div = inputs()
    div = pd.DataFrame([{"record_date": d.date.iloc[1], "ex_date": d.date.iloc[2],
                         "payment_date": d.date.iloc[4], "cash_dividend_per_share": .1}])
    d.loc[2:, ["open", "close"]] = 9.9
    d.loc[2, "dividend"] = .1
    a = weight_account(d, div, p, r, "STRESS", d.date.iloc[1])
    recorded = a["daily"].shares.iloc[0]
    assert a["trades"].dividend_cny.iloc[0] == recorded * .1
    assert a["daily"].dividend_paid.sum() == recorded * .1
    verify_account(a)
