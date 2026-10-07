"""从本地定期报告API原记录重建更正候选目录；不推断修订方向或交易信号。"""
from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd
import yaml

from research import a_share_hs_cninfo_periodic_report_metadata_v1 as metadata
from scripts import run_a_share_hs_cninfo_periodic_report_event_archive_v2_1 as archive


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_t11_correction_catalogue_v1"
CLOCK = ROOT / "reports/research/510300_factor96_t11_clock_preflight_v1"
TOKENS = ("更正", "取消", "修订", "更新", "补充", "差错", "追溯", "重述")


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def report_period_from_title(title):
    """只连接明确的单年单期；多个年度或报告期均保留歧义。"""
    text = metadata.normalize_title(title)
    years = sorted(set(re.findall(r"(?<!\d)(20\d{2})年", text)))
    kinds = []
    if "第一季度报告" in text or "一季度报告" in text:
        kinds.append(("Q1", 3, 31))
    half = any(token in text for token in ("半年度报告", "中期报告", "半年报"))
    if half:
        kinds.append(("H1", 6, 30))
    if "第三季度报告" in text or "三季度报告" in text:
        kinds.append(("Q3", 9, 30))
    # 先移除半年报字样，避免将“半年度报告”误认成年度报告。
    annual_text = text.replace("半年度报告", "").replace("半年报", "").replace("中期报告", "")
    if "年度报告" in annual_text or "年报" in annual_text:
        kinds.append(("FY", 12, 31))
    if len(years) != 1 or len(kinds) != 1:
        return None, None, "NO_VIEW_AMBIGUOUS_OR_MISSING_REPORT_PERIOD"
    kind, month, day = kinds[0]
    return kind, pd.Timestamp(int(years[0]), month, day), "UNIQUE_EXPLICIT_YEAR_AND_REPORT_TYPE"


def candidate_record(raw):
    title = metadata.normalize_title(raw.get("announcementTitle"))
    matched = [token for token in TOKENS if token in title]
    if not matched:
        return None
    kind, period, status = report_period_from_title(title)
    url = str(raw.get("adjunctUrl") or "")
    stamp = metadata.timestamp_at(raw.get("announcementTime"))
    published = metadata.url_publication_date(url)
    date_match = stamp is not None and published is not None and stamp.date() == published.date()
    return {
        "announcement_id": str(raw.get("announcementId") or ""),
        "ts_code": metadata.ts_code_from_cninfo_code(raw.get("secCode")),
        "announcement_title": title, "matched_tokens": "|".join(matched),
        "period_type": kind, "report_period": period, "period_identity_status": status,
        "event_publication_date": published, "official_timestamp_at": stamp,
        "official_internal_date_equal": bool(date_match),
        "relative_pdf_url": url, "official_pdf_url": "https://static.cninfo.com.cn/" + url.lstrip("/"),
        "query_interval": str(raw.get("query_interval") or ""),
        "content_admission": "NOT_READ_NUMERICAL_REVISION_DIRECTION_UNKNOWN",
    }


def main():
    assert not OUT.exists(), "本地更正候选目录已存在；保留既有记录，不覆盖运行。"
    OUT.mkdir(parents=True)
    (OUT / "source_evidence").mkdir()
    save("protocol.json", {
        "at": now(), "study_id": "510300_FACTOR96_T11_CORRECTION_CATALOGUE_V1",
        "scope": "只重建既有307097条定期报告API原记录，筛选全部标题更正候选；不下载原文，不读取价格或修订金额。",
        "title_tokens": list(TOKENS),
        "period_link": "单一明确报告年和单一报告类型、发行人一致、日期内部一致且名义公开日不早于原报告，才列为同报告候选；这不是已证实的修订根链接。",
        "ambiguity": "多年度、多报告期、缺年、缺报告类型、无规范原始报告等分别保留未知；不根据后来的收益或数值挑选。",
        "economic_direction": "标题更正不等于利润下修或原信息失效；退出规则尚未制定，不把候选直接视为卖出信号。",
        "coverage_limit": "原API请求范围是定期报告类别；不能证明覆盖全部临时更正、会计差错或其他会改变原信息的公告。",
        "clock_limit": "沿用官方元数据名义日期；历史first_seen与同一URL的首次版本均未建立。",
        "new_accounts": 0, "new_network_requests": 0, "returns_read": False,
        "goal_achieved": False, "orders_authorized": False,
    })
    sources = [
        archive.CONFIG_PATH,
        ROOT / "research/a_share_hs_cninfo_periodic_report_metadata_v1.py",
        ROOT / "research/a_share_hs_cninfo_periodic_report_event_archive_v2_1.py",
        ROOT / "scripts/run_a_share_hs_cninfo_periodic_report_event_archive_v2_1.py",
        ROOT / "scripts/audit_a_share_hs_cninfo_periodic_report_metadata_v2_0_1_outputs.py",
        ROOT / "tests/test_factor96_t11_correction_catalogue_v1.py",
        CLOCK / "inputs/report_events.parquet", CLOCK / "inputs/membership.parquet", CLOCK / "result.json",
    ]
    config = yaml.safe_load(archive.CONFIG_PATH.read_text(encoding="utf-8"))
    for name in ["base_config", "base_receipt", "first_correction_config", "v2_0_1_config", "v2_0_1_receipt", "v2_0_1_inventory"]:
        sources.append(ROOT / config["references"][name])
    frozen = []
    for i, source in enumerate(sources):
        target = OUT / "source_evidence" / f"{i:02d}_{source.name}"
        shutil.copy2(source, target)
        frozen.append({"source_path": source.relative_to(ROOT).as_posix(), "path": target.relative_to(OUT).as_posix(),
                       "bytes": target.stat().st_size, "sha256": digest(target)})
    shutil.copy2(Path(__file__), OUT / "catalogue_code.py")
    save("input_identity.json", {"at": now(), "files": frozen})
    print("正在只读重建本地公告原记录；不发出网络请求。", flush=True)
    source_receipt = archive.load_json(ROOT / config["references"]["v2_0_1_receipt"])
    inventory = pd.read_csv(ROOT / config["references"]["v2_0_1_inventory"], dtype=str, keep_default_na=False)
    raw, reconstruction = archive.reconstruct_raw_records(config, source_receipt=source_receipt, inventory=inventory)
    assert len(raw) == int(config["raw_reconstruction"]["expected_raw_record_count"])
    assert reconstruction["all_checkpoint_hashes_match"]
    assert reconstruction["duplicate_across_partitions"] == 0 and not reconstruction["identity_conflicts"]
    with gzip.open(OUT / "reconstructed_api_records.jsonl.gz", "xt", encoding="utf-8") as stream:
        for item in raw:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    records = [candidate_record(item) for item in raw]
    candidates = pd.DataFrame([item for item in records if item is not None])
    assert candidates.announcement_id.is_unique
    original = pd.read_parquet(CLOCK / "inputs/report_events.parquet")
    original = original[["ts_code", "report_period", "announcement_id", "event_publication_date"]].rename(columns={
        "announcement_id": "canonical_original_announcement_id", "event_publication_date": "canonical_original_publication_date"})
    original["report_period"] = pd.to_datetime(original.report_period)
    original["canonical_original_publication_date"] = pd.to_datetime(original.canonical_original_publication_date)
    candidates["report_period"] = pd.to_datetime(candidates.report_period)
    candidates["event_publication_date"] = pd.to_datetime(candidates.event_publication_date)
    candidates = candidates.merge(original, on=["ts_code", "report_period"], how="left", validate="many_to_one")
    candidates["candidate_link_status"] = "NO_VIEW_NO_CANONICAL_ORIGINAL_FOR_EXPLICIT_PERIOD"
    unknown_period = candidates.report_period.isna()
    bad_clock = ~candidates.official_internal_date_equal
    before_original = candidates.event_publication_date.lt(candidates.canonical_original_publication_date)
    same_or_after = candidates.canonical_original_announcement_id.notna() & ~before_original & ~bad_clock & ~unknown_period
    candidates.loc[same_or_after, "candidate_link_status"] = "SAME_ISSUER_PERIOD_LATER_OR_SAME_DATE_CANDIDATE_ONLY"
    candidates.loc[before_original, "candidate_link_status"] = "NO_VIEW_CANDIDATE_PRECEDES_CANONICAL_ORIGINAL"
    candidates.loc[bad_clock, "candidate_link_status"] = "NO_VIEW_CLOCK_IDENTITY"
    candidates.loc[unknown_period, "candidate_link_status"] = "NO_VIEW_AMBIGUOUS_OR_MISSING_REPORT_PERIOD"
    members = pd.read_parquet(CLOCK / "inputs/membership.parquet", columns=["symbol"])
    candidates["in_historical_member_union"] = candidates.ts_code.isin(set(members.symbol))
    candidates = candidates.sort_values(["event_publication_date", "ts_code", "announcement_id"], kind="stable")
    candidates.to_parquet(OUT / "correction_candidates.parquet", index=False)
    member_candidates = candidates.loc[candidates.in_historical_member_union]
    member_candidates.to_csv(OUT / "historical_member_correction_candidates.csv", index=False, encoding="utf-8-sig")
    counts = {}
    for name, table in [("all_candidates", candidates), ("historical_member_union_candidates", member_candidates)]:
        counts[name] = {"rows": len(table), "issuers": int(table.ts_code.nunique()),
                        "candidate_link_status": table.candidate_link_status.value_counts().to_dict()}
    result = {
        "at": now(), "status": "LOCAL_CORRECTION_CANDIDATE_CATALOGUE_COMPLETE_CONTENT_NOT_VERIFIED",
        "reconstruction": reconstruction, "counts": counts,
        "source_scope": "历史定期报告类别原始API记录，不是全类别公司公告目录。",
        "financial_values_or_pdf_contents_read": False, "same_period_candidates_are_confirmed_revision_roots": False,
        "content_verification": "NOT_RUN", "complete_information_invalidation_exit_chain": "NOT_ESTABLISHED",
        "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "new_accounts": 0, "new_returns": 0,
        "new_network_requests": 0, "goal_achieved": False, "orders_authorized": False,
    }
    save("result.json", result)
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    save("result_freeze.json", {"at": now(), "files": files})
    print(json.dumps({"状态": result["status"], "统计": counts, "新账户": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
