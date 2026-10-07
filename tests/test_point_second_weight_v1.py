"""候选切换必须改变真实目标、保持未知与时钟、保留来源身份。"""
import numpy as np
import pandas as pd
import pytest

from research.point_account_nr7_complement_v1 import verify_account
from research.point_account_nr7_inputs_v1 import PARENT_A, PARENT_B
from research.point_second_weight_inputs_v1 import source_account, source_parents
from research.point_weight_information_inputs_v1 import weight_account


def inputs():
    dates = pd.bdate_range("2024-01-02", periods=8)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "dividend": 0.})
    parents = pd.DataFrame({"origin": dates, PARENT_A: [.4, 0., 0., 0., 0., 0., 0., 0.],
                            PARENT_B: [.1, .1, np.nan, .4, 0., 0., 0., 0.]})
    risks = pd.DataFrame({"idx": np.arange(8), "es95": .01})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return data, parents, risks, div


def test_second_candidate_uses_own_weight_and_exit_instead_of_first_candidate():
    d, p, r, div = inputs()
    untouched = p.copy(deep=True)
    result = source_account(d, div, p, r, "STRESS", d.date.iloc[1], PARENT_B)
    assert result["orders"].side.tolist() == ["BUY", "BUY", "SELL"]
    assert result["orders"].date.tolist() == [d.date.iloc[1], d.date.iloc[4], d.date.iloc[5]]
    assert result["orders"].quantity.iloc[0] == 2000
    assert result["decisions"].loc[result["decisions"].origin.eq(d.date.iloc[2]), "source_weight"].isna().all()
    assert result["daily"].signal_source.eq(PARENT_B).all()
    assert result["terminal"]["signal_source"] == PARENT_B
    pd.testing.assert_frame_equal(p, untouched)
    verify_account(result)
    short = source_account(d.iloc[:4], div, p, r, "STRESS", d.date.iloc[1], PARENT_B)
    pd.testing.assert_frame_equal(short["daily"], result["daily"].iloc[:3])
    assert short["terminal"]["open_shares"] > 0


def test_adapter_preserves_first_candidate_account_exactly_except_source_annotation():
    d, p, r, div = inputs()
    original = weight_account(d, div, p, r, "STRESS", d.date.iloc[1])
    adapted = source_account(d, div, p, r, "STRESS", d.date.iloc[1], PARENT_A)
    for name in ("daily", "trades", "orders", "decisions", "rejections"):
        if len(original[name].columns) == 0:
            assert adapted[name].empty and adapted[name].columns.tolist() == ["signal_source"]
            continue
        pd.testing.assert_frame_equal(original[name], adapted[name].drop(columns="signal_source"))
    assert adapted["trades"].exit_date.iloc[0] == d.date.iloc[2]


def test_missing_or_invalid_second_candidate_cannot_fall_back_to_first():
    _, p, _, _ = inputs()
    with pytest.raises(ValueError):
        source_parents(p.drop(columns=PARENT_B), PARENT_B)
    with pytest.raises(ValueError):
        source_parents(p, "未登记来源")
    for value in (-.01, 1.01, np.inf):
        invalid = p.copy()
        invalid.loc[0, PARENT_B] = value
        with pytest.raises(ValueError):
            source_parents(invalid, PARENT_B)
