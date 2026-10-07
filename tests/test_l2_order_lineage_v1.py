"""验证剩余委托避免双扣、集合竞价双侧扣减及不确定方向边界。"""

import numpy as np
import pandas as pd

from research.l2_order_lineage_v1 import compare_snapshot_book, inventory_events, pressure_bounds, trade_lineage


def fixtures():
    orders = pd.DataFrame([
        [92959000, 5, "A", "B", 39990, 300],
        [92959000, 10, "A", "S", 40000, 100],
        [92959010, 11, "A", "S", 40010, 500],
        [93000020, 20, "A", "B", 40000, 200],
    ], columns=["time", "ex_order_id", "order_type", "order_code", "price", "volume"])
    trades = pd.DataFrame([[20260709, 93000010, 1001, 10, 20, 40000, 100, "B"]],
                          columns=["date", "time", "trade_id", "ask_order_id", "bid_order_id", "price", "volume", "bs_flag"])
    return orders, trades


def test_aggressive_fill_does_not_get_subtracted_again_from_later_remainder():
    orders, trades = fixtures()
    lineage, links = trade_lineage(orders, trades)
    events, summary = inventory_events(orders, lineage)
    assert links["passive_links_found"] == 1
    assert links["internally_supported_direction_rows"] == 1
    assert summary["negative_remaining_timestamp_batches"] == 0
    assert events.groupby("order_id").delta.sum().loc[20] == 200
    assert events.groupby("order_id").delta.sum().loc[10] == 0


def quote(clock, bids, asks):
    row = {"date": 20260709, "time": clock, "tot_bid_vol": sum(v for _, v in bids), "tot_ask_vol": sum(v for _, v in asks)}
    for side, levels in [("bid", bids), ("ask", asks)]:
        for i in range(1, 11):
            price, size = levels[i - 1] if i <= len(levels) else (0, 0)
            row[f"{side}_px{i}"] = price
            row[f"{side}_vol{i}"] = size
    return row


def test_snapshot_cutoff_does_not_import_future_remainder():
    orders, trades = fixtures()
    lineage, _ = trade_lineage(orders, trades)
    events, _ = inventory_events(orders, lineage)
    snapshots = pd.DataFrame([quote(93000015, [(39990, 300)], [(40010, 500)]),
                              quote(93000030, [(40000, 200), (39990, 300)], [(40010, 500)])])
    result = compare_snapshot_book(snapshots, events)
    assert result.all_ten_prices_and_quantities_match.all()
    assert result.both_total_quantities_match.all()
    assert result.rebuilt_bid_px1.tolist() == [39990, 40000]


def test_auction_consumes_both_orders_without_an_aggressor_claim():
    orders, trades = fixtures()
    orders = orders.loc[orders.ex_order_id.isin([5, 10])].copy()
    orders["time"] = 92000000
    trades.loc[0, ["time", "bid_order_id"]] = [92500000, 5]
    lineage, _ = trade_lineage(orders, trades)
    events, summary = inventory_events(orders, lineage)
    balances = events.groupby("order_id").delta.sum()
    assert balances.loc[5] == 200 and balances.loc[10] == 0
    assert not lineage.same_source_direction_supported.any()
    assert summary["negative_end_balances"] == 0


def test_missing_passive_addition_is_reported_and_not_synthesized():
    orders, trades = fixtures()
    orders = orders.loc[orders.ex_order_id.ne(10)]
    lineage, links = trade_lineage(orders, trades)
    events, summary = inventory_events(orders, lineage)
    assert links["passive_links_found"] == 0
    assert not lineage.same_source_direction_supported.any()
    assert summary["missing_order_reference_events"] == 1
    assert 10 not in events.order_id.tolist()


def test_direction_bounds_include_all_unknown_volume_and_exclude_future():
    rows = pd.DataFrame({"date": [20260709] * 3, "time": [95900000, 95930000, 100001000],
                         "volume": [80.0, 20.0, 999999.0], "same_source_direction_supported": [True, False, True],
                         "supported_direction_sign": [-1, 0, 1]})
    ends = pd.DatetimeIndex(["2026-07-09 10:00"], tz="Asia/Shanghai")
    result = pressure_bounds(rows, ends).iloc[0]
    assert result.total_volume == 100
    assert np.isclose(result.imbalance_lower, -1)
    assert np.isclose(result.imbalance_upper, -.6)
    assert result.net_selling_under_both_extremes
