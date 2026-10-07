"""检验批内所有可行次序被包含，以及窗口外消息不会进入行情范围。"""

from itertools import permutations

import numpy as np
import pandas as pd

from research.l2_order_lineage_v1 import inventory_events, trade_lineage
from research.quote_state_envelope_v1 import batch_outer_bounds, clock_ms, clock_windows, event_comparison, quote_envelopes


def test_clock_arithmetic_across_minute_boundary_is_real_elapsed_time():
    assert (clock_ms([100000000]) - clock_ms([95959990])).item() == 10


def test_outer_bounds_cover_every_feasible_within_batch_permutation():
    start = np.array([100., 50.])
    locations = np.array([0, 0, 1, 1])
    deltas = np.array([80., -140., 20., -30.])
    lower, upper, end = batch_outer_bounds(start, locations, deltas)
    for order in permutations(range(len(deltas))):
        state = start.copy()
        path = [state.copy()]
        for i in order:
            state[locations[i]] += deltas[i]
            path.append(state.copy())
        if min(float(value.min()) for value in path) < 0:
            continue
        for value in path:
            assert np.all(value >= lower) and np.all(value <= upper)
        assert np.array_equal(state, end)


def test_clock_interval_respects_next_excluded_trade_batch():
    trades = pd.DataFrame({"time": [93000100, 93000300], "volume": [100., 100.]})
    quotes = pd.DataFrame({"time": [93000000], "cum_volume": [100.]})
    row = clock_windows(quotes, trades).iloc[0]
    assert row.lower_ms == clock_ms([93000100]).item()
    assert row.upper_exclusive_ms == clock_ms([93000300]).item() + 10


def test_future_message_is_outside_envelope_and_observed_snapshot_is_contained():
    messages = []
    for i in range(10):
        messages.append([92959000, i + 1, "A", "B", 39990 - i * 10, 300])
    for i in range(11):
        messages.append([92959000, i + 11, "A", "S", 40000 + i * 10, 100])
    messages.extend([[93000020, 30, "A", "B", 40000, 200], [93001010, 31, "A", "B", 50000, 99999]])
    orders = pd.DataFrame(messages, columns=["time", "ex_order_id", "order_type", "order_code", "price", "volume"])
    trades = pd.DataFrame([[20260709, 93000010, 1001, 11, 30, 40000, 100, "B"]], columns=["date", "time", "trade_id", "ask_order_id", "bid_order_id", "price", "volume", "bs_flag"])
    lineage, _ = trade_lineage(orders, trades)
    events, _ = inventory_events(orders, lineage)
    quote = {"date": 20260709, "time": 93000000, "cum_volume": 100, "tot_bid_vol": 3200, "tot_ask_vol": 1000}
    for i in range(1, 11):
        quote[f"bid_px{i}"] = 40010 - i * 10
        quote[f"bid_vol{i}"] = 200 if i == 1 else 300
        quote[f"ask_px{i}"] = 40000 + i * 10
        quote[f"ask_vol{i}"] = 100
    row = quote_envelopes(pd.DataFrame([quote]), trades, events).iloc[0]
    assert row.source_snapshot_inside_outer_box
    assert row.mid_lower_cny <= 4.0005 <= row.mid_upper_cny
    assert row.mid_upper_cny < 4.01
    assert not row.availability_at_nominal_time_proven


def test_direction_is_unknown_when_price_bounds_cross_zero():
    start = pd.Series({"feature_status": "CONDITIONAL_OUTER_ENVELOPE", "mid_lower_cny": 4., "mid_upper_cny": 4.002,
                       "bid_depth_lower_cny": 0., "bid_depth_upper_cny": 10000., "spread_lower_ticks": 1., "spread_upper_ticks": 2.})
    end = start.copy()
    result = event_comparison(start, end)
    assert result["price_direction"] == "ZERO_OR_DIRECTION_AMBIGUOUS"
    assert np.isnan(result["moving_bid_depth_ratio_upper"])


def test_source_outside_envelope_cannot_be_promoted_to_robust_observation():
    row = pd.Series({"source_snapshot_inside_outer_box": False, "feature_status": "CONDITIONAL_OUTER_ENVELOPE"})
    assert event_comparison(row, row)["comparison_status"] == "SOURCE_OUTSIDE_CONDITIONAL_BOUND_NO_INFERENCE"
