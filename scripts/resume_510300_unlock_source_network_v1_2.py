"""连接恢复后只补三个声明月份一次，不修改已有采集结果。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.repair_510300_unlock_announcement_source_v1_1 import check, merge_records, now, repair_month, save, sha

OLD = ROOT/"reports/research/510300_unlock_announcement_source_v1_1/source_repair_result.json"
REPORT = ROOT/"reports/research/510300_unlock_announcement_source_v1_2"
PROTOCOL = ROOT/"docs/510300_UNLOCK_ANNOUNCEMENT_SOURCE_NETWORK_RESUMPTION_V1_2.md"
MONTHS = ["2015-01", "2015-05", "2015-08"]


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    check(old["status"] == "NO_VIEW_REPAIR_INCOMPLETE", "V1.1未结束为预期不完整状态")
    oldout = ROOT/old["raw_output"]
    coverage = pd.read_csv(oldout/"month_coverage.csv").to_dict("records")
    failed = [row for row in coverage if not row["status"].startswith("PASS_")]
    check(sorted(row["month"] for row in failed) == MONTHS, "失败月份不等于声明的三个连接失败月份")
    check(all("10053" in row["error"] and "Connection aborted" in row["error"] for row in failed), "失败类型不是已声明连接中断")
    check(sum(row["status"] == "PASS_REPAIRED_DECLARED_MONTH" for row in coverage) >= 30, "缺少连接恢复后真实成功月份")
    oldframe = pd.read_parquet(oldout/"all_announcement_records.parquet")
    check(sha(oldout/"all_announcement_records.parquet") == old["output_files"]["all_announcement_records.parquet"]["sha256"], "V1.1合并文件变化")
    cfgpath = ROOT/"config/510300_unlock_announcement_source_v1_1.json"
    config = json.loads(cfgpath.read_text(encoding="utf-8"))
    config.update(max_http_attempts_per_page=1, workers=1)
    original = json.loads((ROOT/config["original_config"]).read_text(encoding="utf-8"))
    originalout = ROOT/config["original_output"]
    original_result = json.loads((ROOT/config["original_result"]).read_text(encoding="utf-8"))
    check(sha(originalout/"all_announcement_records.parquet") == original_result["output_files"]["all_announcement_records.parquet"]["sha256"], "原V1公告表哈希变化")
    originalframe = pd.read_parquet(originalout/"all_announcement_records.parquet")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_0800")
    output = ROOT/"data/raw/market/510300_unlock_announcements_v1_2"/stamp
    protected = [Path(__file__), PROTOCOL, OLD, cfgpath, ROOT/"scripts/repair_510300_unlock_announcement_source_v1_1.py", ROOT/config["original_config"], ROOT/"scripts/collect_510300_unlock_announcements_v1.py"]
    save(REPORT/"network_resumption_freeze.json", {"status": "FROZEN_SOURCE_NETWORK_RESUMPTION_BEFORE_LABELS", "frozen_at": now(), "months": MONTHS, "new_max_attempts_per_page": 1, "max_additional_batches": 1, "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in protected], "new_return_labels": 0, "new_models": 0, "new_accounts": 0})
    save(REPORT/"network_resumption_claim.json", {"started_at": now(), "pid": os.getpid(), "output": output.relative_to(ROOT).as_posix()})
    output.mkdir(parents=True, exist_ok=False)
    rows = oldframe.to_dict("records")
    summaries = [row for row in coverage if row["month"] not in MONTHS]
    for month in MONTHS:
        previous = json.loads((originalout/month/"month_receipt.json").read_text(encoding="utf-8"))
        summary, recovered = repair_month(previous, output, config, original)
        if summary["status"] == "PASS_REPAIRED_DECLARED_MONTH":
            summary["status"] = "PASS_NETWORK_RESUMED_DECLARED_MONTH"
        summaries.append(summary)
        rows.extend(recovered)
        print(f"{month} {summary['status']} {len(recovered)}条", flush=True)
    all_rows = merge_records([rows], len(originalframe))
    frame = pd.DataFrame(all_rows)[list(originalframe.columns)]
    combined = merge_records([originalframe.to_dict("records"), all_rows], len(originalframe))
    check(len(combined) <= len(originalframe), "旧新身份并集超过原声明")
    new_ids, old_ids = set(frame.announcementId), set(originalframe.announcementId)
    difference = {"original_rows": len(originalframe), "original_unique_ids": len(old_ids), "final_unique_ids": len(new_ids), "added_ids": sorted(new_ids-old_ids), "missing_original_ids": sorted(old_ids-new_ids)}
    frame.to_parquet(output/"all_announcement_records.parquet", index=False)
    frame.loc[frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE")].to_parquet(output/"eligible_original_disclosures.parquet", index=False)
    frame.loc[frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY")].to_parquet(output/"revision_titles.parquet", index=False)
    summaries.sort(key=lambda row: row["month"])
    pd.DataFrame(summaries).to_csv(output/"month_coverage.csv", index=False, encoding="utf-8-sig")
    save(output/"difference_from_original.json", difference)
    good = all(row["status"].startswith("PASS_") for row in summaries) and len(frame) == len(originalframe) and not difference["missing_original_ids"]
    result = {"source_study_id": "510300_UNLOCK_ANNOUNCEMENT_SOURCE_V1_2", "status": "PASS_COMPLETE_PRIMARY_ANNOUNCEMENT_METADATA_AFTER_EXPLICIT_SOURCE_REPAIR" if good else "NO_VIEW_NETWORK_RESUMPTION_INCOMPLETE_STOP", "completed_at": now(), "raw_output": output.relative_to(ROOT).as_posix(), "requested_months": 140, "complete_months": sum(row["status"].startswith("PASS_") for row in summaries), "original_status": "NO_VIEW_INCOMPLETE_PRIMARY_ANNOUNCEMENT_COVERAGE", "v1_1_status": old["status"], "original_result": config["original_result"], "v1_1_result": OLD.relative_to(ROOT).as_posix(), "v1_1_result_sha256": sha(OLD), "all_unique_records": len(frame), "additional_recovered_ids_vs_original": len(difference["added_ids"]), "eligible_original_records": int(frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE").sum()), "revision_records": int(frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY").sum()), "additional_http_attempts": sum(row["http_attempts"] for row in summaries if row["month"] in MONTHS), "additional_failed_http_attempts": sum(row["failed_attempts"] for row in summaries if row["month"] in MONTHS), "new_return_labels": 0, "new_models": 0, "new_accounts": 0, "goal_achieved": False, "historical_intraday_first_delivery_proven": False, "network_budget_explicitly_amended": True, "further_retry_batches_allowed": 0, "output_files": {name: {"sha256": sha(output/name), "bytes": (output/name).stat().st_size} for name in ["all_announcement_records.parquet", "eligible_original_disclosures.parquet", "revision_titles.parquet", "month_coverage.csv", "difference_from_original.json"]}}
    save(output/"completion.json", result)
    save(REPORT/"source_completion_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not good:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
