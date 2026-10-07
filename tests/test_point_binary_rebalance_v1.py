"""验证删除仓位大小信息没有删除未知状态或改变事前成交边界。"""
import numpy as np
import pandas as pd
import pytest

from research.point_account_nr7_complement_v1 import verify_account
from research.point_account_nr7_inputs_v1 import PARENT_A, PARENT_B
from research.point_binary_rebalance_inputs_v1 import binary_rebalance_parents
from research.point_weight_information_inputs_v1 import weight_account


def test_projection_preserves_unknown_zero_other_source_and_prefix():
    parents = pd.DataFrame({"origin": pd.bdate_range("2024-01-02", periods=5),
                            PARENT_A: [0., .01, .8, np.nan, 1.], PARENT_B: .2})
    original = parents.copy(deep=True)
    result = binary_rebalance_parents(parents)
    np.testing.assert_allclose(result[PARENT_A], [0., .5, .5, np.nan, .5], equal_nan=True)
    pd.testing.assert_frame_equal(parents, original)
    pd.testing.assert_series_equal(result[PARENT_B], original[PARENT_B])
    pd.testing.assert_frame_equal(binary_rebalance_parents(parents.iloc[:4]), result.iloc[:4])


def test_projection_does_not_turn_invalid_targets_or_duplicate_dates_into_signals():
    for value in (-.01, 1.01, np.inf, -np.inf, "坏值"):
        parents = pd.DataFrame({"origin": [pd.Timestamp("2024-01-02")], PARENT_A: [value]})
        with pytest.raises(ValueError):
            binary_rebalance_parents(parents)
    parents = pd.DataFrame({"origin": [pd.Timestamp("2024-01-02")] * 2, PARENT_A: [0., .1]})
    with pytest.raises(ValueError):
        binary_rebalance_parents(parents)


def test_risk_reduction_and_replenishment_keep_one_cycle_and_next_open_execution():
    dates = pd.bdate_range("2024-01-02", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "dividend": 0.})
    parents = pd.DataFrame({"origin": dates, PARENT_A: [.1] * 7 + [0.] * 3})
    risks = pd.DataFrame({"idx": np.arange(10), "es95": [.01, .01, .20] + [.01] * 7})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    projected = binary_rebalance_parents(parents)
    result = weight_account(data, dividends, projected, risks, "STRESS", dates[1])
    orders = result["orders"]
    assert orders.loc[orders.date.eq(dates[3]), "side"].tolist() == ["SELL"]
    assert orders.loc[orders.date.eq(dates[4]), "side"].tolist() == ["BUY"]
    assert orders.iloc[-1]["date"] == dates[8]
    assert orders.origin.lt(orders.date).all()
    assert len(result["trades"]) == 1
    assert result["trades"].status.iloc[0] == "COMPLETE"
    assert result["daily"].exposure.max() <= .5 + 1e-12
    verify_account(result)
    shortened = weight_account(data.iloc[:6], dividends, projected, risks, "STRESS", dates[1])
    pd.testing.assert_frame_equal(result["daily"].iloc[:5], shortened["daily"])
    assert shortened["terminal"]["open_shares"] > 0
