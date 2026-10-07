"""检查本批历史子集覆盖、保存完整性和重叠日期收盘价，不计算策略收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo_history_20261001"
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/historical_expansion_20261002"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    summary = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "ALL_TARGET_SOURCE_SUBSETS_SAVED":
        raise SystemExit("本批下载尚未全部完成，保留当前进度，不生成全期覆盖结论。")
    registration = json.loads((BASE / "registration.json").read_text(encoding="utf-8"))
    manifest = json.loads((BASE / "coverage_manifest.json").read_text(encoding="utf-8"))
    reference_probe_path = BASE / "reference_symbol_presence_20260930.json"
    reference_probe = json.loads(reference_probe_path.read_text(encoding="utf-8")) if reference_probe_path.exists() else None
    if OUT.exists():
        raise SystemExit("本批覆盖结果已存在，拒绝覆盖。")
    minute_path = ROOT / "data/raw/510300_free_channels_v1/20261001/neigezhu/data/etf_1m/SH/510300.parquet"
    minute = pd.read_parquet(minute_path, columns=["timestamp", "close"])
    timestamps = pd.to_datetime(minute.timestamp)
    closes = minute.loc[timestamps.dt.strftime("%H:%M").eq("15:00"), ["close"]].copy()
    closes["date"] = timestamps.loc[closes.index].dt.strftime("%Y%m%d")
    if closes.date.duplicated().any():
        raise ValueError("分钟对照存在重复收盘日期。")
    close_map = closes.set_index("date").close.to_dict()
    days = registration["dates"]
    by_day = {day: {"date": day, "regime": "PRE_20260706" if day < "20260706" else "POST_20260706"} for day in days}
    prior_by_regime = {"PRE_20260706": 0, "POST_20260706": 0}
    for day in days:
        row = by_day[day]
        prior_count = prior_by_regime[row["regime"]]
        row["prior_available_source_days_same_regime"] = prior_count
        row["source_day_count_at_least_60_not_qualified_baseline"] = prior_count >= 60
        prior_by_regime[row["regime"]] += 1
    seen = set()
    for item in manifest["files"]:
        key = (item["date"], item["stream"])
        if key in seen:
            raise ValueError("覆盖清单包含重复的日期和流。")
        seen.add(key)
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError("已保存原件哈希与提取回执不一致。")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != item["rows"] or parquet.schema_arrow.names != item["columns"]:
            raise ValueError("本地Parquet行数或列与提取回执不一致。")
        row = by_day[item["date"]]
        stream = item["stream"]
        row[stream + "_rows"] = item["rows"]
        row[stream + "_bytes"] = item["bytes"]
        row[stream + "_first_time"] = item["first_time"]
        row[stream + "_last_time"] = item["last_time"]
        row[stream + "_rows_at_or_after_1505"] = item["rows_at_or_after_1505"]
        if stream == "行情":
            row["positive_iopv_rows"] = item.get("positive_iopv_rows", 0)
            quotes = parquet.read(columns=["date", "time", "price"]).to_pandas()
            close_candidates = quotes.loc[quotes.time.ge(145959000) & quotes.time.le(150100000) & quotes.price.gt(0)]
            observed = float(close_candidates.sort_values("time", kind="stable").iloc[-1].price / 10000) if len(close_candidates) else None
            reference = float(close_map[item["date"]]) if item["date"] in close_map else None
            row["source_close_cny"] = observed
            row["minute_source_close_cny"] = reference
            row["close_difference_ticks"] = (observed - reference) / .001 if observed is not None and reference is not None else None
            row["close_comparison_status"] = "MATCH" if row["close_difference_ticks"] is not None and abs(row["close_difference_ticks"]) < 1e-6 else "DIFFERENCE_RETAINED" if row["close_difference_ticks"] is not None else "NO_OVERLAPPING_CLOSE"
    expected = {(day, stream) for day in days for stream in registration["config"]["streams"]}
    if seen != expected:
        raise ValueError("文件覆盖与已登记完整日期区间不一致。")
    frame = pd.DataFrame([by_day[day] for day in days])
    OUT.mkdir(parents=True, exist_ok=False)
    frame.to_csv(OUT / "01_逐日三流覆盖与收盘对照.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "SOURCE_SUBSET_COVERAGE_AND_SAVED_FILES_VERIFIED",
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_repo": manifest["repo"], "source_revision": manifest["revision"],
        "target_date_count": len(days), "complete_three_stream_dates": len(days),
        "verified_files": len(seen), "date_start": days[0], "date_end": days[-1],
        "source_rows_by_stream": {stream: int(frame[stream + "_rows"].sum()) for stream in registration["config"]["streams"]},
        "rows_total": int(sum(frame[stream + "_rows"].sum() for stream in registration["config"]["streams"])),
        "raw_subset_bytes_including_reused": int(sum(frame[stream + "_bytes"].sum() for stream in registration["config"]["streams"])),
        "new_directory_bytes": sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()),
        "new_network_body_bytes": summary["new_network_body_bytes"],
        "additional_reference_symbol_probe_bytes": reference_probe["network_body_bytes"] if reference_probe else 0,
        "new_network_body_bytes_including_reference_probe": summary["new_network_body_bytes"] + (reference_probe["network_body_bytes"] if reference_probe else 0),
        "reference_symbol_presence_on_20260930": reference_probe["symbol_row_counts"] if reference_probe else None,
        "reused_files": summary["reused_files"],
        "regime_day_counts": {key: int(value) for key, value in frame.regime.value_counts().items()},
        "source_history_count_only": {
            regime: {
                "days_with_at_least_60_prior_source_days": int(part.source_day_count_at_least_60_not_qualified_baseline.sum()),
                "first_day_with_60_prior_source_days": str(part.loc[part.source_day_count_at_least_60_not_qualified_baseline, "date"].iloc[0]) if part.source_day_count_at_least_60_not_qualified_baseline.any() else None,
                "qualified_same_hhmm_baseline_proven": False,
            }
            for regime, part in frame.groupby("regime", sort=True)
        },
        "positive_iopv_day_count": int(frame.positive_iopv_rows.gt(0).sum()),
        "positive_iopv_rows": int(frame.positive_iopv_rows.sum()),
        "close_comparison_counts": {key: int(value) for key, value in frame.close_comparison_status.value_counts().items()},
        "close_comparison_is_independent_order_or_receipt_validation": False,
        "source_message_completeness_proven": False,
        "historical_receive_clock_proven": False,
        "strategy_returns_computed": False, "actual_orders": 0, "goal_achieved": False,
        "provenance": [
            {"path": str(BASE / "registration.json"), "sha256": digest(BASE / "registration.json")},
            {"path": str(BASE / "coverage_manifest.json"), "sha256": digest(BASE / "coverage_manifest.json")},
            {"path": str(minute_path), "sha256": digest(minute_path)},
            {"path": str(Path(__file__)), "sha256": digest(Path(__file__))},
        ],
    }
    save_json("summary.json", payload)
    save_json("goal_progress.json", {
        "status": "HISTORICAL_ACQUISITION_RESUMED_WITH_USER_CLARIFICATION",
        "previous_local_only_interpretation_corrected": True,
        "new_evidence": "已按完整交易日区间取得并检查181天510300三流原值文件，不能再以本地只有三天为由停止历史研究。",
        "old_blocked_evidence_is_historical": True,
        "original_profit_goal_achieved": False,
        "next_step": "分别使用119天旧制度、62天新制度资料核对时段和字段，继续构建历史测量；同步估值与成交条件另行判断。",
    })
    index = [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(OUT.iterdir()) if path.is_file()]
    pd.DataFrame(index).to_csv(OUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
