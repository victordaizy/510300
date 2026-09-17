"""从原始分页重核公告身份及成员覆盖；不读取行情或收益标签。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_510300_unlock_announcements_v1 import clean_title, title_kind
from scripts.repair_510300_unlock_announcement_source_v1_1 import check, CORE_KEYS

RESULT = ROOT/"reports/research/510300_unlock_announcement_source_admission_v1/daily_source_consolidation.json"
OUT = ROOT/"reports/research/510300_unlock_announcement_source_admission_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}


def reconstruct() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = json.loads((ROOT/"config/510300_unlock_announcement_source_v1.json").read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    check(result["status"] == "PASS_COMPLETE_DAILY_DOCUMENT_METADATA_WITH_RECORDED_INTRADAY_TIMESTAMP_DRIFT", "来源未完整，禁止构造成分覆盖")
    directory = ROOT/result["raw_output"]
    for name, info in result["output_files"].items():
        check(sha(directory/name) == info["sha256"], "来源完成文件哈希变化："+name)
    frame = pd.read_parquet(directory/"all_announcement_records.parquet")
    check(len(frame) == result["all_unique_records"] == 33722, "公告行数与固定声明不同")
    check(not frame.announcementId.duplicated().any(), "公告编号重复")
    coverage = pd.read_csv(directory/"month_coverage.csv")
    check(len(coverage) == 140 and coverage.month.nunique() == 140, "来源月覆盖不完整")
    check(coverage.status.str.startswith("PASS_").all(), "月份存在未完成状态")
    counts = frame.groupby("source_month").size()
    for row in coverage.itertuples():
        check(int(counts[row.month]) == int(row.declared_records), "月份公告量与声明量不同")
    raw_checked, attempts = 0, []
    for path, group in frame.groupby("raw_response_path", sort=True):
        raw = ROOT/path
        expected = set(group.raw_response_sha256)
        check(len(expected) == 1 and sha(raw) in expected, "逐行原始响应哈希不匹配")
        payload = json.loads(raw.read_text(encoding="utf-8"))
        raw_rows = payload["announcements"] or []
        lookup = {str(row["announcementId"]): row for row in raw_rows}
        check(len(lookup) == len(raw_rows), "实际准入的单响应内部有重复")
        receipt = json.loads(raw.with_name(raw.stem+"_receipt.json").read_text(encoding="utf-8"))
        check(receipt["status"] == "HTTP_AND_SCHEMA_PASS" and receipt["tls_verify"] is True and receipt["http_status"] == 200, "准入原始响应缺少成功HTTP回执")
        check(receipt["sha256"] == sha(raw), "原始响应与HTTP回执不一致")
        check(receipt["body"]["searchkey"] == cfg["search_keyword"], "检索词变化")
        for item in group.to_dict("records"):
            original = lookup.get(str(item["announcementId"]))
            check(original is not None and all(original.get(key) == item[key] for key in CORE_KEYS), "归档表不能从响应重建公告身份")
            clock = pd.to_datetime(int(original["announcementTime"]), unit="ms", utc=True).tz_convert("Asia/Shanghai")
            check(clock.isoformat() == item["announcement_timestamp_shanghai"] and clock.strftime("%Y-%m-%d") == item["announcement_date"], "公告时间转换不一致")
            check(clean_title(original["announcementTitle"]) == item["clean_title"], "清理标题不一致")
            check(title_kind(original["announcementTitle"], cfg) == item["title_kind"], "标题准入规则不一致")
        raw_checked += 1
        attempts.append({"raw_path": path, "rows_admitted_from_response": len(group), "sha256": sha(raw)})
    members_path = ROOT/cfg["historical_membership"]
    check(sha(members_path) == cfg["historical_membership_sha256"], "历史成员名单版本不符")
    members = pd.read_parquet(members_path)
    members["membership_date"] = pd.to_datetime(members.membership_date)
    check(not members.duplicated(["membership_date", "symbol"]).any(), "成员日期和证券键重复")
    check(members.groupby("membership_date").size().eq(300).all(), "逐日成分不是300只")
    sessions = pd.DatetimeIndex(sorted(members.membership_date.unique()))
    check(len(members) == 846900 and len(sessions) == 2823, "既有成员输入长度不符")
    eligible = frame.loc[frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE")].copy()
    dates = pd.to_datetime(eligible.announcement_date)
    positions = sessions.searchsorted(dates)
    check((positions < len(sessions)).all(), "公告日期在成员覆盖之后")
    eligible["membership_diagnostic_session"] = sessions[positions].to_numpy()
    code = eligible.secCode.astype(str)
    eligible["symbol"] = np.where(code.str.startswith("6"), code+".SH", np.where(code.str.startswith(("0", "3")), code+".SZ", code+".OTHER"))
    marker = members[["membership_date", "symbol"]].rename(columns={"membership_date": "membership_diagnostic_session"}).copy()
    marker["is_csi300_at_diagnostic_session"] = True
    eligible = eligible.merge(marker, on=["membership_diagnostic_session", "symbol"], how="left", validate="many_to_one")
    eligible["is_csi300_at_diagnostic_session"] = eligible.is_csi300_at_diagnostic_session.eq(True)
    company_days = eligible.groupby(["announcement_date", "secCode", "symbol", "membership_diagnostic_session"], as_index=False).agg(announcement_count=("announcementId", "size"), is_csi300=("is_csi300_at_diagnostic_session", "any"))
    eligible["year"] = eligible.announcement_date.str[:4]
    company_days["year"] = company_days.announcement_date.str[:4]
    years = []
    for year, group in frame.groupby(frame.announcement_date.str[:4]):
        e = eligible.loc[eligible.year.eq(year)]
        c = company_days.loc[company_days.year.eq(year)]
        years.append({"year": year, "all_query_announcements": len(group), "eligible_original_announcements": len(e), "eligible_company_days": len(c), "csi300_eligible_announcements": int(e.is_csi300_at_diagnostic_session.sum()), "csi300_eligible_company_days": int(c.is_csi300.sum()), "csi300_distinct_issuers": c.loc[c.is_csi300, "secCode"].nunique(), "csi300_activity_dates": c.loc[c.is_csi300, "announcement_date"].nunique()})
    midnight = frame.announcement_timestamp_shanghai.str[11:19].eq("00:00:00")
    path_date = frame.adjunctUrl.str.extract(r"finalpage/(\d{4}-\d{2}-\d{2})/", expand=False)
    diagnostic = {"all_records": len(frame), "raw_responses_rechecked": raw_checked, "eligible_original_announcements": len(eligible), "eligible_company_days": len(company_days), "csi300_eligible_announcements": int(eligible.is_csi300_at_diagnostic_session.sum()), "csi300_eligible_company_days": int(company_days.is_csi300.sum()), "csi300_unique_issuers": int(company_days.loc[company_days.is_csi300, "secCode"].nunique()), "midnight_placeholder_rows": int(midnight.sum()), "non_midnight_metadata_rows": int((~midnight).sum()), "archive_path_date_mismatches": int(frame.announcement_date.ne(path_date).sum()), "title_classes": {str(k): int(v) for k,v in frame.title_kind.value_counts().items()}, "membership_rows": len(members), "membership_sessions": len(sessions), "membership_sha256": sha(members_path), "membership_mapping_is_diagnostic_not_trade_availability": True, "historical_weights_used": False, "historical_intraday_first_delivery_proven": False, "response_provenance": attempts}
    return diagnostic, eligible, company_days, pd.DataFrame(years)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "verify"])
    args = parser.parse_args()
    diagnostic, eligible, company_days, years = reconstruct()
    if args.mode == "build":
        OUT.mkdir(parents=True, exist_ok=True)
        check(not (OUT/"source_admission.json").exists(), "来源准入已存在，不能覆盖")
        eligible.to_parquet(OUT/"eligible_with_membership_diagnostic.parquet", index=False)
        company_days.to_parquet(OUT/"company_day_activity_diagnostic.parquet", index=False)
        years.to_csv(OUT/"year_coverage.csv", index=False, encoding="utf-8-sig")
        provenance = diagnostic.pop("response_provenance")
        save(OUT/"admitted_response_provenance.json", provenance)
        report = {"study_id": "510300_UNLOCK_ANNOUNCEMENT_SOURCE_ADMISSION_V1", "status": "PASS_SOURCE_METADATA_AND_PIT_MEMBERSHIP_DIAGNOSTIC_ONLY", "completed_at": now(), "source_completion": identity(RESULT), **diagnostic, "return_experiment": "NOT_RUN_REQUIRES_SEPARATE_FROZEN_PROTOCOL", "new_return_labels": 0, "new_models": 0, "new_accounts": 0, "goal_achieved": False, "position_impact": 0, "verifier": identity(Path(__file__)), "outputs": [identity(OUT/name) for name in ["eligible_with_membership_diagnostic.parquet", "company_day_activity_diagnostic.parquet", "year_coverage.csv", "admitted_response_provenance.json"]]}
        save(OUT/"source_admission.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        saved = json.loads((OUT/"source_admission.json").read_text(encoding="utf-8"))
        check(sha(Path(__file__)) == saved["verifier"]["sha256"], "来源重核代码变化")
        for item in saved["outputs"]:
            check(sha(ROOT/item["path"]) == item["sha256"], "保存来源输出变化")
        diagnostic.pop("response_provenance")
        for key, value in diagnostic.items():
            check(value == saved[key], "重算来源诊断不同："+key)
        pd.testing.assert_frame_equal(eligible, pd.read_parquet(OUT/"eligible_with_membership_diagnostic.parquet"))
        pd.testing.assert_frame_equal(company_days, pd.read_parquet(OUT/"company_day_activity_diagnostic.parquet"))
        print(json.dumps({"status": "PASS_OFFLINE_RAW_SOURCE_AND_MEMBERSHIP_RECOMPUTATION", "verified_at": now(), "all_records_checked": diagnostic["all_records"], "raw_responses_checked": diagnostic["raw_responses_rechecked"], "membership_sessions": diagnostic["membership_sessions"], "new_network_calls": 0, "new_return_labels": 0, "new_models": 0, "new_accounts": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
