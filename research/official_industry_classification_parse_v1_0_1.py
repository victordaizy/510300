"""仅从首轮已保存的八份解析结果完成案例汇总，修复空成员归组异常。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research import broker_fixed_cohort_observation_v1 as cohort
from research import official_industry_classification_inputs_v1 as inputs
from research import official_industry_publication_intake_v1 as intake

ROOT, ORIGINAL = intake.ROOT, intake.OUT
OUT = ORIGINAL / "implementation_v1_0_1"


def case_tables(decision, source_date, members, metadata, snapshot_loader):
    """空源日保持未知；失败快照不回退；行业分组只接受已知成员。"""
    decision, source_date = pd.Timestamp(decision), pd.Timestamp(source_date)
    if members.symbol.duplicated().any():
        raise ValueError("案例前一源日成员重复，不能合并。")
    observed_at = (decision + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai")
    published = [row for row in metadata if inputs.publication_clock(row["published"]) <= observed_at]
    latest = max(published, key=lambda row: (inputs.publication_clock(row["published"]), row["id"])) if published else None
    chosen = inputs.choose_snapshot(metadata, observed_at)
    membership_known = len(members) == 300
    snapshot = snapshot_loader(chosen) if chosen is not None and membership_known else None
    classified = inputs.classify_members(members, snapshot)
    classified["decision_date"], classified["membership_source_date"] = decision, source_date
    known = int(classified.classification_known.sum())
    view = ("NO_VIEW_MEMBERSHIP_SOURCE_MISSING" if not membership_known else
        "NO_VIEW_LATEST_PUBLISHED_SNAPSHOT_PARSE_FAILED" if latest is not None and not latest["parse_passed"] else
        "NO_VIEW_NO_PUBLISHED_SNAPSHOT" if chosen is None else
        "PARTIAL_MEMBER_CLASSIFICATION_UNKNOWN" if known < len(members) else "CASE_MEMBERS_CLASSIFIED")
    case = {"decision_date": decision, "membership_source_date": source_date,
        "latest_published_snapshot_id": latest["id"] if latest else None,
        "latest_published_parse_passed": latest["parse_passed"] if latest else None,
        "snapshot_id": chosen["id"] if chosen else None,
        "snapshot_publication_date": chosen["published"] if chosen else None,
        "snapshot_available_at": inputs.publication_clock(chosen["published"]) if chosen else pd.NaT,
        "taxonomy": chosen["taxonomy"] if chosen else None,
        "members": len(members), "membership_view_allowed": membership_known,
        "classified_members": known,
        "unknown_members": len(members) - known if membership_known else None,
        "coverage": known / len(members) if membership_known else None,
        "unknown_symbols": ";".join(classified.loc[~classified.classification_known, "symbol"]),
        "clock_passed": chosen is None or inputs.publication_clock(chosen["published"]) <= observed_at,
        "view_state": view}
    industries = []
    if snapshot is not None and known:
        for key, group in classified.loc[classified.classification_known].groupby("industry_key"):
            industries.append({"decision_date": decision, "snapshot_id": chosen["id"], "taxonomy": chosen["taxonomy"],
                "industry_key": key, "major_name": group.major_name.iloc[0], "member_count": len(group),
                "count_share_of_300": len(group) / 300, "meaning": "成员数量占比，不是沪深300指数权重或产业因果"})
    return case, classified, industries


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("空成员隔离修复已登记，不能覆盖。")
    received = intake.parent.read(ORIGINAL / "summary.json")
    tests = intake.parent.read(OUT / "tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 2:
        raise ValueError("空成员及未知快照两个必要回归尚未通过。")
    paths = [Path(__file__), ROOT / "tests/test_official_industry_classification_v1_0_1.py",
        OUT / "tests_receipt.json", ORIGINAL / "parser_protocol.json", ORIGINAL / "PARSER_STARTED.json",
        ORIGINAL / "initial_parser_failure_receipt.json"]
    original = intake.parent.read(ORIGINAL / "parser_protocol.json")
    paths.extend(ROOT / item["path"] for item in original["files"])
    for node in received["nodes"]:
        folder = ORIGINAL / "parsed" / node["id"]
        paths.extend(folder / name for name in ("all_pages.txt", "raw_table_rows.parquet", "classification.parquet", "parse_result.json"))
    paths = list(dict.fromkeys(paths))
    intake.parent.write(OUT / "protocol.json", {"at": intake.parent.original.now(),
        "registration": "TECH.R217_SOURCE_IMPLEMENTATION_EMPTY_MEMBERSHIP_FIX", "decision": "TECH.R218",
        "purpose": "FINISH_NINE_SOURCE_CASES_FROM_SAVED_EIGHT_ORIGINAL_PARSES",
        "only_change": "成员源缺失时仍保存案例未知记录，不对缺少industry_key的空表归组。",
        "preserved": "原8 PDF、702页解析、表格原行、5通过/3失败与5未知分类、9案例/源日、公布钟及原六测试逐值保持；不推测缺失代码、不回退旧分类。",
        "new_gets": 0, "new_pdf_parses": 0, "new_accounts": 0, "new_fits": 0, "new_labels": 0,
        "files": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": intake.parent.digest(path)} for path in paths]})
    print("空成员案例汇总修复已固定：复用8已保存解析，0重新请求/解析/账户。", flush=True)


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("本隔离汇总已开始，不重复运行。")
    protocol = intake.parent.read(OUT / "protocol.json")
    for item in protocol["files"]:
        if intake.parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("隔离修复登记后的原文件改变：" + item["path"])
    intake.parent.write(OUT / "RUN_STARTED.json", {"at": intake.parent.original.now(), "new_accounts": 0})
    received = intake.parent.read(ORIGINAL / "summary.json")
    parsed = [intake.parent.read(ORIGINAL / "parsed" / node["id"] / "parse_result.json") for node in received["nodes"]]
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(cohort.SOURCEFILES["observed"], columns=["date"]).date))
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("原完整日历不是严格升序且唯一。")
    membership = pd.read_parquet(cohort.SOURCEFILES["membership"])
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    cases, member_frames, industries = [], [], []
    for date in intake.CASE_DATES:
        decision = pd.Timestamp(date)
        if decision not in calendar or calendar.get_loc(decision) == 0:
            raise ValueError("固定案例没有原完整前一交易日。")
        source_date = calendar[calendar.get_loc(decision) - 1]
        members = membership.loc[membership.membership_date.eq(source_date), ["symbol"]].copy()
        case, frame, groups = case_tables(decision, source_date, members, parsed,
            lambda chosen: pd.read_parquet(ROOT / chosen["classification_path"]))
        cases.append(case)
        member_frames.append(frame)
        industries.extend(groups)
    unknown = []
    for node in parsed:
        frame = pd.read_parquet(ROOT / node["classification_path"])
        unknown.append(frame.loc[~frame.classification_known].copy())
    table("八快照全部解析质量与源钟", pd.DataFrame(parsed))
    table("九固定案例公布钟与300成员分类覆盖", pd.DataFrame(cases))
    table("九案例全部已取得成员_已知与未知保留", pd.concat(member_frames, ignore_index=True))
    table("九案例全部已知行业_成员数量非指数权重", pd.DataFrame(industries))
    table("原件全部未知分类_保持原五证券行", pd.concat(unknown, ignore_index=True))
    changed = [item["path"] for item in protocol["files"] if intake.parent.digest(ROOT / item["path"]) != item["sha256"]]
    if changed:
        raise ValueError("复用原解析时固定输入改变：" + ";".join(changed))
    summary = {"at": intake.parent.original.now(), "decision": "TECH.R218", "registration": "TECH.R217",
        "status": "SELECTED_OFFICIAL_SOURCE_CASE_COVERAGE_COMPLETED_NOT_FULL_FINANCIAL_ADMISSION",
        "original_partial_failure_retained": True, "original_parsed_tables_preserved": True,
        "implementation": "v1_0_1_empty_membership_summary_only",
        "parsed": parsed, "pdfs_received": received["pdfs_received"],
        "parse_passed_snapshots": sum(row["parse_passed"] for row in parsed),
        "parse_failed_snapshots": sum(not row["parse_passed"] for row in parsed),
        "logical_gets": received["logical_gets"], "pages_read": sum(row["pages"] for row in parsed),
        "classification_rows": sum(row["rows"] for row in parsed),
        "unknown_classification_rows": sum(row["unknown_classification_rows"] for row in parsed),
        "case_rows": cases, "case_member_rows": sum(len(frame) for frame in member_frames),
        "industry_count_rows": len(industries), "original_tests_passed": 6, "necessary_regressions_passed": 2,
        "new_gets_in_implementation": 0, "new_pdf_parses_in_implementation": 0,
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "financial_metrics": "NOT_COMPUTED", "financial_admission": "NOT_ESTABLISHED_EIGHT_CASE_SNAPSHOTS_NOT_FULL_HISTORY",
        "first_vintage": "NOT_CERTIFIED_CURRENT_RETRIEVAL_OFFICIAL_FILES", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False, "source_files_unchanged": len(protocol["files"])}
    intake.parent.write(OUT / "summary.json", summary)
    print(f"9案例完成，{summary['case_member_rows']}原成员行，原5通过/3失败和5未知分类保持；0新金融。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复用官方分类原解析，仅修复空成员案例归组异常。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
