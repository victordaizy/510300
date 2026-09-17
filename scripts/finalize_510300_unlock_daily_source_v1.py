"""离线合并完整公告日档案，保留一秒级接口时间变化和旧失败。"""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.repair_510300_unlock_announcement_source_v1_1 import check, now, save, sha

REPORT = ROOT/"reports/research/510300_unlock_announcement_source_admission_v1"
OUT = ROOT/"data/curated/510300_unlock_announcement_daily_source_v1"
PROTOCOL = ROOT/"docs/510300_UNLOCK_ANNOUNCEMENT_DAILY_SOURCE_ADMISSION_V1.md"


def main():
    first_path = ROOT/"reports/research/510300_unlock_announcement_source_v1/source_collection_result.json"
    second_path = ROOT/"reports/research/510300_unlock_announcement_source_v1_1/source_repair_result.json"
    recovery_report = ROOT/"reports/research/510300_unlock_announcement_source_v1_2"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    recovery = json.loads((recovery_report/"network_resumption_claim.json").read_text(encoding="utf-8"))
    failure = json.loads((recovery_report/"failure_receipt.json").read_text(encoding="utf-8"))
    check(failure["status"] == "PROGRAM_FAILED_AFTER_SUCCESSFUL_THREE_MONTH_NETWORK_CAPTURE", "网络批次失败回执状态不符")
    a, b, c = ROOT/first["raw_output"], ROOT/second["raw_output"], ROOT/recovery["output"]
    for directory, result in [(a, first), (b, second)]:
        check(sha(directory/"all_announcement_records.parquet") == result["output_files"]["all_announcement_records.parquet"]["sha256"], "保存原始归档表哈希变化")
    original = pd.read_parquet(a/"all_announcement_records.parquet")
    repaired = pd.read_parquet(b/"all_announcement_records.parquet")
    rows = repaired.to_dict("records")
    coverage = pd.read_csv(b/"month_coverage.csv").to_dict("records")
    incomplete = [x["month"] for x in coverage if not x["status"].startswith("PASS_")]
    check(sorted(incomplete) == ["2015-01", "2015-05", "2015-08"], "待合并月份与声明不符")
    coverage = [x for x in coverage if x["month"] not in incomplete]
    for month in sorted(incomplete):
        receipt = json.loads((c/month/"month_receipt.json").read_text(encoding="utf-8"))
        check(receipt["status"] == "PASS_REPAIRED_DECLARED_MONTH", "网络恢复月份未完整")
        source_rows = json.loads((c/month/"records.json").read_text(encoding="utf-8"))
        check(len(source_rows) == receipt["declared_records"], "恢复月份记录量不符")
        rows.extend(source_rows)
        receipt["status"] = "PASS_NETWORK_MONTH_DATA_UNDER_SEPARATE_DAILY_ADMISSION"
        coverage.append(receipt)
    frame = pd.DataFrame(rows)[list(original.columns)]
    frame = frame.sort_values(["announcement_date", "announcementId"]).reset_index(drop=True)
    check(len(frame) == len(original) == 33722 and not frame.announcementId.duplicated().any(), "最终编号数与原声明量不符")
    lookup = frame.set_index("announcementId").to_dict("index")
    old_ids, new_ids = set(original.announcementId), set(frame.announcementId)
    check(old_ids.issubset(new_ids), "合并丢失原已见编号")
    differences = []
    core = ["secCode", "orgId", "announcementTitle", "adjunctUrl", "announcement_date"]
    for item in original.drop_duplicates("announcementId").to_dict("records"):
        new = lookup[item["announcementId"]]
        check(all(item[key] == new[key] for key in core), "公告日或文档身份变化")
        if item["announcementTime"] != new["announcementTime"]:
            old_clock = pd.to_datetime(int(item["announcementTime"]), unit="ms", utc=True).tz_convert("Asia/Shanghai")
            new_clock = pd.to_datetime(int(new["announcementTime"]), unit="ms", utc=True).tz_convert("Asia/Shanghai")
            check(old_clock.date() == new_clock.date(), "时间变化跨过公告日")
            differences.append({"announcementId": item["announcementId"], "secCode": item["secCode"], "title": item["clean_title"], "announcement_date": item["announcement_date"], "original_clock": old_clock.isoformat(), "selected_repaired_clock": new_clock.isoformat(), "delta_seconds": (new_clock-old_clock).total_seconds(), "original_raw": item["raw_response_path"], "original_raw_sha256": item["raw_response_sha256"], "repaired_raw": new["raw_response_path"], "repaired_raw_sha256": new["raw_response_sha256"], "disposition": "DATE_ONLY_ADMITTED_INTRADAY_FIRST_DELIVERY_NOT_PROVEN"})
    check(len(differences) == 1 and differences[0]["announcementId"] == "1225030021" and differences[0]["delta_seconds"] == 1, "差异不是已记录的唯一一秒变化")
    protected = [Path(__file__), PROTOCOL, first_path, second_path, recovery_report/"failure_receipt.json", a/"all_announcement_records.parquet", b/"all_announcement_records.parquet"]
    save(REPORT/"daily_scope_freeze.json", {"status": "FROZEN_DAILY_SOURCE_CONSOLIDATION_BEFORE_LABELS", "frozen_at": now(), "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in protected], "new_network_calls": 0, "new_return_labels": 0, "new_models": 0, "new_accounts": 0})
    OUT.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(OUT/"all_announcement_records.parquet", index=False)
    frame.loc[frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE")].to_parquet(OUT/"eligible_original_disclosures.parquet", index=False)
    frame.loc[frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY")].to_parquet(OUT/"revision_titles.parquet", index=False)
    coverage.sort(key=lambda x: x["month"])
    pd.DataFrame(coverage).to_csv(OUT/"month_coverage.csv", index=False, encoding="utf-8-sig")
    save(OUT/"timestamp_observation_differences.json", differences)
    save(OUT/"difference_from_original.json", {"original_returned_rows": len(original), "original_unique_ids": len(old_ids), "final_unique_ids": len(new_ids), "newly_recovered_ids": sorted(new_ids-old_ids), "missing_old_ids": []})
    result = {"source_study_id": "510300_UNLOCK_ANNOUNCEMENT_DAILY_SOURCE_V1", "status": "PASS_COMPLETE_DAILY_DOCUMENT_METADATA_WITH_RECORDED_INTRADAY_TIMESTAMP_DRIFT", "completed_at": now(), "raw_output": OUT.relative_to(ROOT).as_posix(), "complete_months": len(coverage), "start_date": "2015-01-01", "end_date": "2026-08-14", "all_unique_records": len(frame), "newly_recovered_ids": len(new_ids-old_ids), "recorded_timestamp_conflicts": len(differences), "maximum_recorded_clock_difference_seconds": 1, "old_failed_statuses_preserved": True, "historical_intraday_first_delivery_proven": False, "admitted_temporal_granularity": "SHANGHAI_CALENDAR_ANNOUNCEMENT_DATE_ONLY", "new_network_calls": 0, "new_return_labels": 0, "new_models": 0, "new_accounts": 0, "goal_achieved": False, "output_files": {name: {"sha256": sha(OUT/name), "bytes": (OUT/name).stat().st_size} for name in ["all_announcement_records.parquet", "eligible_original_disclosures.parquet", "revision_titles.parquet", "month_coverage.csv", "timestamp_observation_differences.json", "difference_from_original.json"]}}
    save(REPORT/"daily_source_consolidation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
