"""累计成交量只能证明已包含正量成交，不对缺失前缀插值。"""

import pandas as pd

from scripts.diagnose_510300_quote_clock_v1 import locate_volume_prefix


def test_exact_volume_prefix_exposes_subsecond_trade_after_nominal_quote():
    trades = pd.DataFrame({"date": [20260709] * 3, "time": [93000100, 93000300, 93001000], "volume": [100., 200., 100.]})
    quotes = pd.DataFrame({"date": [20260709] * 2, "time": [93000000, 93001000], "cum_volume": [300., 400.]})
    result = locate_volume_prefix(quotes, trades)
    assert result.positive_trade_prefix_exact.all()
    assert result.contains_trade_later_than_nominal_quote.tolist() == [True, False]
    assert result.included_trade_minus_nominal_quote_seconds.tolist() == [.3, 0.]


def test_unmatched_volume_and_zero_quantity_do_not_create_a_false_time_bound():
    trades = pd.DataFrame({"date": [20260709] * 3, "time": [93000100, 93000300, 93001000], "volume": [100., 0., 200.]})
    quotes = pd.DataFrame({"date": [20260709] * 2, "time": [93000000, 93001000], "cum_volume": [100., 250.]})
    result = locate_volume_prefix(quotes, trades)
    assert result.included_trade_minus_nominal_quote_seconds.iloc[0] == .1
    assert not result.positive_trade_prefix_exact.iloc[1]
    assert pd.isna(result.latest_included_trade_time_lower_bound.iloc[1])
