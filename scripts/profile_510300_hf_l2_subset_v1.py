"""复核已提取的510300原值子集；仅诊断覆盖和字段，不生成策略收益。"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001"
REPORT = ROOT / "reports/research/510300_pressure_recovery_v1/free_channels_20261001/l2_subset_review"
STREAMS = ("行情", "逐笔委托", "逐笔成交")
CN = timezone(timedelta(hours=8))


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不能直接序列化类型：{type(value).__name__}")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar, allow_nan=False) + "\n", encoding="utf-8")


def time_text(value):
    text = f"{int(value):09d}"
    return f"{text[:2]}:{text[2:4]}:{text[4:6]}.{text[6:]}"


def milliseconds(series):
    value = series.astype("int64")
    return (value // 10000000) * 3600000 + ((value // 100000) % 100) * 60000 + ((value // 1000) % 100) * 1000 + value % 1000


def counts(series):
    return {str(key): int(value) for key, value in series.fillna("<NULL>").value_counts(dropna=False).items()}


def describe_file(frame, receipt):
    times = frame.time.astype("int64")
    valid_clock = (times // 10000000 < 24) & ((times // 100000) % 100 < 60) & ((times // 1000) % 100 < 60)
    return {
        "source_path": receipt["source_path"],
        "source_sha256_is_full_day_file": False,
        "subset_sha256": receipt["sha256"],
        "saved_bytes": receipt["bytes"],
        "rows": len(frame),
        "source_codes": sorted(frame.wind_code.unique().tolist()),
        "source_dates": sorted(frame.date.unique().tolist()),
        "first_time": time_text(times.min()),
        "last_time": time_text(times.max()),
        "invalid_clock_rows": int((~valid_clock).sum()),
        "rows_after_1505": int(times.ge(150500000).sum()),
        "rows_during_1505_to_1530": int(times.between(150500000, 153000000, inclusive="both").sum()),
        "all_field_duplicate_rows": int(frame.duplicated().sum()),
        "null_fields": {c: int(frame[c].isna().sum()) for c in frame if frame[c].isna().any()},
        "categorical_fields": {c: counts(frame[c]) for c in ("trade_flag", "trade_code", "order_type", "order_code", "bs_flag") if c in frame},
    }


def quote_diagnostics(frame):
    times = frame.time
    morning = times.ge(93000000) & times.le(113000000)
    afternoon = times.ge(130000000) & times.lt(145700000)
    continuous = frame.loc[morning | afternoon]
    bid = continuous.bid_px1
    ask = continuous.ask_px1
    valid = bid.gt(0) & ask.gt(bid) & np.isfinite(bid) & np.isfinite(ask)
    gaps = []
    for mask in (morning, afternoon):
        observed = milliseconds(frame.loc[mask, "time"]).sort_values()
        gaps.extend((observed.diff().dropna() / 1000).tolist())
    mid = (ask + bid) / 2
    band_complete = valid & continuous.bid_px10.gt(0) & continuous.ask_px10.gt(0)
    band_complete &= continuous.bid_px10.le(mid * (1 - 10 / 10000))
    band_complete &= continuous.ask_px10.ge(mid * (1 + 10 / 10000))
    book_violations = 0
    for level in range(1, 10):
        a, b = continuous[f"ask_px{level}"], continuous[f"ask_px{level+1}"]
        c, d = continuous[f"bid_px{level}"], continuous[f"bid_px{level+1}"]
        book_violations += int(((a.gt(0) & b.gt(0) & b.lt(a)) | (c.gt(0) & d.gt(0) & d.gt(c))).sum())
    return {
        "iopv_zero_rows": int(frame.iopv.eq(0).sum()),
        "iopv_positive_rows": int(frame.iopv.gt(0).sum()),
        "iopv_null_rows": int(frame.iopv.isna().sum()),
        "iopv_scale_verified": False,
        "zero_iopv_interpretation": "不可用参考估值，不能当作真实净值为零，也不以前后日净值补齐。",
        "continuous_rows": len(continuous),
        "valid_positive_uncrossed_best_quotes": int(valid.sum()),
        "invalid_or_locked_or_crossed_best_quotes": int((~valid).sum()),
        "adjacent_level_order_violations": book_violations,
        "ten_levels_cover_both_sides_10bps_band_rows": int(band_complete.sum()),
        "continuous_snapshot_gap_seconds_median": float(np.median(gaps)) if gaps else None,
        "continuous_snapshot_gap_seconds_max": float(max(gaps)) if gaps else None,
        "continuous_snapshot_gaps_over_5_seconds": sum(value > 5 for value in gaps),
        "feed_receive_timestamp_available": False,
        "quote_clock_error_independently_verified": False,
        "after_hours_quote_fields_are_validated_order_queue": False,
    }


def order_trade_diagnostics(orders, trades):
    exchange_ids = set(orders.loc[orders.ex_order_id.gt(0), "ex_order_id"].astype("uint64").tolist())
    ids = set(orders.loc[orders.order_id.gt(0), "order_id"].astype("uint64").tolist())
    result = {
        "zero_order_id_rows": int(orders.order_id.eq(0).sum()),
        "zero_ex_order_id_rows": int(orders.ex_order_id.eq(0).sum()),
        "order_id_join": {},
        "ex_order_id_join": {},
        "missing_join_is_not_proof_of_missing_exchange_messages": True,
        "join_limitation": "沪市委托可能只发布即时成交后的剩余量；缺少订单行不能直接判定丢包，也不能凭成交总量重建完整排队。",
        "aggressor_direction_independently_validated": False,
    }
    for column in ("ask_order_id", "bid_order_id"):
        observed = trades.loc[trades[column].gt(0), column]
        for key, pool in (("order_id_join", ids), ("ex_order_id_join", exchange_ids)):
            result[key][column] = {"nonzero_trade_links": len(observed), "links_found_in_orders": int(observed.isin(pool).sum())}
    return result


def reconciliation(quotes, trades, minute_day):
    result = {"price_scale_cny_divisor_candidate": 10000, "price_scale_source": "仓库字段说明；本报告另做同日分钟和量额的原值核对。", "quantity_scale_assumed_for_comparison_only": 1, "amount_scale_assumed_for_comparison_only": 1, "independent_exchange_completeness_proven": False}
    for name, upper in (("regular_close", 150100000), ("including_after_hours", 154000000)):
        q = quotes.loc[quotes.time.lt(upper)].sort_values("time")
        t = trades.loc[trades.time.lt(upper)]
        if q.empty or t.empty:
            result[name] = {"status": "NO_COMPARABLE_ROWS"}
            continue
        last = q.iloc[-1]
        volume = float(t.volume.sum())
        notional = float((t.price / 10000 * t.volume).sum())
        result[name] = {
            "last_quote_time": time_text(last.time),
            "trade_rows": len(t),
            "trade_volume_raw_sum": volume,
            "quote_cum_volume_raw": float(last.cum_volume),
            "volume_difference_raw": volume - float(last.cum_volume),
            "trade_price_div10000_times_raw_volume_sum": notional,
            "quote_cum_amount_raw": float(last.cum_amount),
            "amount_difference_under_candidate_units": notional - float(last.cum_amount),
            "last_quote_price_div10000": float(last.price / 10000),
        }
    if minute_day.empty:
        result["minute_reference"] = {"status": "NO_SAME_DAY_MINUTE_FILE"}
    else:
        last = minute_day.sort_values("timestamp").iloc[-1]
        result["minute_reference"] = {"status": "SAME_DAY_FILE_PRESENT", "rows": len(minute_day), "last_timestamp": str(last.timestamp), "close_cny": float(last.close), "volume_raw_sum": float(minute_day.volume.sum()), "turnover_raw_sum": float(minute_day.turnover.sum()), "regular_close_price_matches": bool(abs(result["regular_close"]["last_quote_price_div10000"] - float(last.close)) < 1e-9)}
    return result


def main():
    if (REPORT / "review.json").exists():
        raise SystemExit("本批诊断已经保存；需要新批次时使用新的报告目录，避免覆盖。")
    receipts = []
    for path in sorted((BASE / "venvoo_selected").glob("*/receipt.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] == "EXTRACTED_PENDING_SEMANTIC_VALIDATION":
            receipts.append((path, record))
    if not receipts:
        raise SystemExit("没有已完成的三流提取批次。")
    minutes = pd.read_parquet(BASE / "neigezhu/data/etf_1m/SH/510300.parquet")
    minute_dates = pd.to_datetime(minutes.timestamp).dt.strftime("%Y%m%d")
    reviews = []
    for receipt_path, receipt in receipts:
        files = {}
        frames = {}
        for stream in STREAMS:
            entry = next(item for item in receipt["files"] if item["source_path"].endswith("/" + stream + ".parquet"))
            path = receipt_path.parent / (stream + ".parquet")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise SystemExit("已存子集哈希与回执不符：" + str(path))
            frame = pd.read_parquet(path)
            if not frame.wind_code.str.fullmatch(r"510300(?:\.[A-Za-z]+)?").all() or set(frame.date.tolist()) != {int(receipt["date"])}:
                raise SystemExit("子集证券或日期不符：" + str(path))
            frames[stream] = frame
            files[stream] = describe_file(frame, entry)
        reviews.append({
            "date": receipt["date"],
            "canonical_instrument": "510300.SH",
            "raw_suffix_is_not_exchange_identity": True,
            "receipt_path": str(receipt_path.resolve()),
            "network_body_bytes": receipt["network_body_bytes"],
            "files": files,
            "quotes": quote_diagnostics(frames["行情"]),
            "order_trade_links": order_trade_diagnostics(frames["逐笔委托"], frames["逐笔成交"]),
            "reconciliation": reconciliation(frames["行情"], frames["逐笔成交"], minutes.loc[minute_dates.eq(receipt["date"])]),
            "admission": {"m1": "NOT_ADMITTED", "m2": "NOT_ADMITTED", "actual_fills_verified": False, "complete_account_returns_computed": False, "missing": ["独立核验的同步参考估值及其误差范围", "经济时点与源端接收时钟证据", "经过验证的可成交数量与自身队列或实际订单回执", "M2同制度此前60个合格交易日基线"]},
        })
        print(f"已核对{receipt['date']}：{sum(len(frame) for frame in frames.values())}条原值记录。", flush=True)
    summary = {
        "reviewed_at": datetime.now(CN).isoformat(),
        "status": "L2_SUBSETS_OBTAINED_WITH_KNOWN_MEASUREMENT_GAPS",
        "repo": receipts[0][1]["repo"],
        "revision": receipts[0][1]["revision"],
        "dates": [item["date"] for item in reviews],
        "total_rows": sum(f["rows"] for item in reviews for f in item["files"].values()),
        "subset_parquet_bytes": sum(f["saved_bytes"] for item in reviews for f in item["files"].values()),
        "successful_runs_network_body_bytes": sum(item["network_body_bytes"] for item in reviews),
        "source_records_are_complete_exchange_feed": False,
        "all_extracted_quote_iopv_zero": all(item["quotes"]["iopv_zero_rows"] == item["files"]["行情"]["rows"] for item in reviews),
        "m1_m2_net_returns": "NOT_COMPUTED",
        "goal_achieved": False,
        "daily_reviews": reviews,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / "review.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "daily_reviews"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
