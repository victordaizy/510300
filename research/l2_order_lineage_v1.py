"""上海ETF剩余委托口径下的同源关联与快照核对，不提供成交保证。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.pressure_recovery_local_exploration_v1 import with_clock
from research.pressure_recovery_measurement_correction_v1_1 import trailing_values


def classify_phase(times: pd.Series) -> np.ndarray:
    values = times.to_numpy()
    return np.select([values < 93000000, values < 145700000, values < 150500000],
                     ["OPEN_AUCTION", "CONTINUOUS_SOURCE", "CLOSE_AUCTION"], default="AFTER_HOURS_UNSUPPORTED")


def trade_lineage(orders: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    additions = orders.loc[orders.order_type.eq("A") & orders.ex_order_id.gt(0)].copy()
    if additions.ex_order_id.duplicated().any():
        raise ValueError("同日新增委托号不唯一，不能用一对一映射。")
    additions = additions.set_index("ex_order_id")
    result = trades[["date", "time", "trade_id", "ask_order_id", "bid_order_id", "price", "volume", "bs_flag"]].copy()
    result["phase"] = classify_phase(result.time)
    result["id_implied_bs"] = np.where(result.bid_order_id.gt(result.ask_order_id), "B", np.where(result.ask_order_id.gt(result.bid_order_id), "S", "U"))
    result["passive_order_id"] = np.minimum(result.ask_order_id, result.bid_order_id)
    result["aggressive_order_id"] = np.maximum(result.ask_order_id, result.bid_order_id)
    result["passive_side_candidate"] = np.where(result.id_implied_bs.eq("B"), "S", np.where(result.id_implied_bs.eq("S"), "B", "U"))
    for key, name in [("passive_order_id", "passive"), ("aggressive_order_id", "aggressive")]:
        for field in ("time", "order_code", "price", "volume"):
            result[f"{name}_add_{field}"] = result[key].map(additions[field])
    continuous = result.phase.eq("CONTINUOUS_SOURCE")
    found = result.passive_add_time.notna()
    result["passive_link_found"] = found
    result["passive_side_consistent"] = found & result.passive_add_order_code.eq(result.passive_side_candidate)
    result["passive_price_consistent"] = found & result.passive_add_price.eq(result.price)
    result["passive_strictly_prior"] = found & result.passive_add_time.lt(result.time)
    result["same_source_direction_supported"] = (continuous & result.passive_strictly_prior & result.passive_side_consistent
                                                   & result.passive_price_consistent
                                                   & (result.aggressive_add_time.isna() | result.aggressive_add_time.ge(result.time)))
    result["supported_direction_sign"] = np.where(result.same_source_direction_supported, np.where(result.id_implied_bs.eq("B"), 1, -1), 0)
    regular = result.loc[continuous]
    summary = {
        "trade_rows": len(result), "source_bs_matches_order_id_comparison": int(result.bs_flag.eq(result.id_implied_bs).sum()),
        "continuous_source_rows": len(regular), "continuous_source_volume": float(regular.volume.sum()),
        "passive_links_found": int(regular.passive_link_found.sum()),
        "passive_link_volume_fraction": float(regular.loc[regular.passive_link_found, "volume"].sum() / regular.volume.sum()) if len(regular) else None,
        "passive_add_later_than_trade": int(regular.passive_add_time.gt(regular.time).sum()),
        "passive_add_same_timestamp": int(regular.passive_add_time.eq(regular.time).sum()),
        "passive_side_mismatches": int((regular.passive_link_found & ~regular.passive_side_consistent).sum()),
        "passive_price_mismatches": int((regular.passive_link_found & ~regular.passive_price_consistent).sum()),
        "aggressive_add_absent": int(regular.aggressive_add_time.isna().sum()),
        "aggressive_add_strictly_earlier": int(regular.aggressive_add_time.lt(regular.time).sum()),
        "internally_supported_direction_rows": int(regular.same_source_direction_supported.sum()),
        "internally_supported_direction_volume_fraction": float(regular.loc[regular.same_source_direction_supported, "volume"].sum() / regular.volume.sum()) if len(regular) else None,
        "independent_aggressor_validation": False,
    }
    return result, summary


def inventory_events(orders: pd.DataFrame, lineage: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按时间批次维护已公布的剩余库存，不从未来新增数量倒扣主动成交。"""
    additions = orders.loc[orders.order_type.eq("A") & orders.ex_order_id.gt(0)].set_index("ex_order_id")
    cancels = orders.loc[orders.order_type.eq("D")].copy()
    parts = []
    for message_type, source, ids, signs in [
        ("ADD", additions.reset_index(), additions.index.to_numpy(), 1),
        ("CANCEL", cancels, cancels.ex_order_id.to_numpy(), -1),
    ]:
        frame = pd.DataFrame({"time": source.time.to_numpy(), "order_id": ids, "delta": signs * source.volume.to_numpy(), "kind": message_type})
        parts.append(frame)
    continuous = lineage.loc[lineage.phase.eq("CONTINUOUS_SOURCE")]
    parts.append(pd.DataFrame({"time": continuous.time.to_numpy(), "order_id": continuous.passive_order_id.to_numpy(), "delta": -continuous.volume.to_numpy(), "kind": "PASSIVE_FILL"}))
    auctions = lineage.loc[lineage.phase.isin(["OPEN_AUCTION", "CLOSE_AUCTION"])]
    for key in ("ask_order_id", "bid_order_id"):
        parts.append(pd.DataFrame({"time": auctions.time.to_numpy(), "order_id": auctions[key].to_numpy(), "delta": -auctions.volume.to_numpy(), "kind": "AUCTION_FILL"}))
    events = pd.concat(parts, ignore_index=True)
    events["side"] = events.order_id.map(additions.order_code)
    events["price"] = events.order_id.map(additions.price)
    events["add_time"] = events.order_id.map(additions.time)
    missing = events.side.isna() | events.price.isna()
    early = events.add_time.gt(events.time)
    valid = events.loc[~missing & ~early].copy()
    grouped = valid.groupby(["order_id", "time"], sort=True).delta.sum().reset_index()
    grouped["remaining_after_timestamp_batch"] = grouped.groupby("order_id", sort=False).delta.cumsum()
    end = grouped.groupby("order_id", sort=False).tail(1)
    summary = {"additions": len(additions), "cancellations": len(cancels), "inventory_events": len(events),
               "missing_order_reference_events": int(missing.sum()), "event_before_add_time": int(early.sum()),
               "negative_remaining_timestamp_batches": int(grouped.remaining_after_timestamp_batch.lt(-1e-6).sum()),
               "negative_end_balances": int(end.remaining_after_timestamp_batch.lt(-1e-6).sum()),
               "zero_end_balances": int(end.remaining_after_timestamp_batch.abs().le(1e-6).sum()),
               "positive_end_balances": int(end.remaining_after_timestamp_batch.gt(1e-6).sum()),
               "after_hours_trade_rows_not_reconstructed": int(lineage.phase.eq("AFTER_HOURS_UNSUPPORTED").sum()),
               "timestamp_batch_does_not_prove_within_batch_order": True}
    return valid.sort_values("time", kind="stable").reset_index(drop=True), summary


def compare_snapshot_book(quotes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """零时间平移；快照时点前全部源消息的库存与源十档逐档比较。"""
    selected = quotes.loc[((quotes.time >= 93000000) & (quotes.time < 113000000)) | ((quotes.time >= 130000000) & (quotes.time < 145700000))].copy()
    selected = selected.loc[selected.bid_px1.gt(0) & selected.ask_px1.gt(selected.bid_px1)].sort_values("time", kind="stable")
    price_grid = np.sort(events.price.astype("int64").unique())
    locations = np.searchsorted(price_grid, events.price.astype("int64").to_numpy())
    bid_events = events.side.eq("B").to_numpy()
    changes = events.delta.to_numpy(dtype=float)
    times = events.time.to_numpy()
    bid, ask = np.zeros(len(price_grid)), np.zeros(len(price_grid))
    cursor = 0
    records = []
    for quote in selected.itertuples():
        stop = int(np.searchsorted(times, quote.time, side="right"))
        local = slice(cursor, stop)
        is_bid = bid_events[local]
        indices, amounts = locations[local], changes[local]
        np.add.at(bid, indices[is_bid], amounts[is_bid])
        np.add.at(ask, indices[~is_bid], amounts[~is_bid])
        cursor = stop
        bid_indices = np.flatnonzero(bid > 1e-6)[::-1][:10]
        ask_indices = np.flatnonzero(ask > 1e-6)[:10]
        record = {"date": int(quote.date), "time": int(quote.time), "source_quote_index": int(quote.Index),
                  "negative_price_level_count": int((bid < -1e-6).sum() + (ask < -1e-6).sum()),
                  "rebuilt_total_bid_shares": float(bid.sum()), "rebuilt_total_ask_shares": float(ask.sum()),
                  "source_total_bid_shares": float(quote.tot_bid_vol), "source_total_ask_shares": float(quote.tot_ask_vol)}
        comparisons = []
        for side, selected_indices, inventory in [("bid", bid_indices, bid), ("ask", ask_indices, ask)]:
            for i in range(1, 11):
                raw_price = int(price_grid[selected_indices[i - 1]]) if i <= len(selected_indices) else 0
                quantity = float(inventory[selected_indices[i - 1]]) if i <= len(selected_indices) else 0.0
                price_matches = raw_price == int(getattr(quote, f"{side}_px{i}"))
                quantity_matches = abs(quantity - float(getattr(quote, f"{side}_vol{i}"))) <= 1e-6
                record[f"rebuilt_{side}_px{i}"] = raw_price
                record[f"rebuilt_{side}_vol{i}"] = quantity
                comparisons.append((price_matches, quantity_matches))
        record["both_best_prices_match"] = comparisons[0][0] and comparisons[10][0]
        record["all_ten_prices_match"] = all(item[0] for item in comparisons)
        record["all_ten_prices_and_quantities_match"] = all(p and q for p, q in comparisons)
        record["both_total_quantities_match"] = abs(record["rebuilt_total_bid_shares"] - record["source_total_bid_shares"]) <= 1e-6 and abs(record["rebuilt_total_ask_shares"] - record["source_total_ask_shares"]) <= 1e-6
        records.append(record)
    return pd.DataFrame(records)


def pressure_bounds(lineage: pd.DataFrame, endpoints: pd.DatetimeIndex) -> pd.DataFrame:
    """对同源时序未能支持方向的全部成交量给出两个极端，而非强行补标。"""
    clocked = with_clock(lineage)
    quantities = clocked.volume.to_numpy(dtype=float)
    supported = clocked.same_source_direction_supported.to_numpy()
    signed = quantities * clocked.supported_direction_sign.to_numpy(dtype=float)
    total = trailing_values(clocked, endpoints, quantities)
    unknown = trailing_values(clocked, endpoints, np.where(supported, 0.0, quantities))
    certain_difference = trailing_values(clocked, endpoints, signed)
    lower = np.divide(certain_difference - unknown, total, out=np.full(len(total), np.nan), where=total > 0)
    upper = np.divide(certain_difference + unknown, total, out=np.full(len(total), np.nan), where=total > 0)
    return pd.DataFrame({"endpoint": endpoints, "total_volume": total, "unknown_direction_volume": unknown,
                         "internally_supported_signed_volume": certain_difference, "imbalance_lower": lower,
                         "imbalance_upper": upper, "net_selling_under_both_extremes": upper < 0,
                         "net_buying_under_both_extremes": lower > 0, "independent_direction_validation": False})
