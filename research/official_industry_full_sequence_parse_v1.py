"""解析全序列新原件并汇总3488日逐行来源覆盖；不计算新收益信号。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pdfplumber

from research import broker_fixed_cohort_observation_v1 as cohort
from research import official_industry_classification_inputs_v1 as original
from research import official_industry_full_sequence_inputs_v1 as inputs
from research import official_industry_full_sequence_intake_v1 as intake

ROOT, OUT, parent = intake.ROOT, intake.OUT, intake.parent


def table(name, frame, csv=True):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    if csv:
        frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "parser_protocol.json").exists():
        raise RuntimeError("全序列解析用途已经登记，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["passed"] != 3 or tests["exit_code"] != 0 or tests["inputs_sha256"] != parent.digest(Path(inputs.__file__)):
        raise ValueError("三个逐行来源必要测试尚未通过或代码改变。")
    sources = parent.read(OUT / "summary.json")
    paths = [Path(__file__), Path(inputs.__file__), Path(original.__file__),
        ROOT / "tests/test_official_industry_full_sequence_v1.py", OUT / "tests_receipt.json",
        OUT / "protocol.json", OUT / "summary.json", cohort.SOURCEFILES["membership"], cohort.SOURCEFILES["observed"]]
    for node in sources["nodes"]:
        if node["pdf_received"]:
            paths.append(ROOT / node["pdf_path"])
            if node["source_reused"]:
                folder = intake.previous.OUT / "parsed" / node["id"]
                paths.extend(folder / name for name in ("parse_result.json", "classification.parquet", "raw_table_rows.parquet", "all_pages.txt"))
    parent.write(OUT / "parser_protocol.json", {"at": parent.original.now(), "registration": intake.REGISTRATION,
        "decision": intake.RESULT, "scope": "NEW25_OR_LESS_RECEIVED_PDFS_ONCE_REUSE8_PARSES_FULL3488_SOURCE_COVERAGE",
        "row_contract": parent.read(OUT / "protocol.json")["row_source_eligible"],
        "old_complete_snapshot_failures": "三个原完整快照失败保持；新用途逐行有效不改原裁决。",
        "clock": "实际公布日23:59，最新结构失败或缺原件不回退，成员用前一完整ETF日。",
        "source_age": "逐日实报年龄和大于365日标记，仅诊断，不按结果引入交易源龄阈值。",
        "tables": "全部槽、所有实际成员含未知、全部行业数量、全年度分母、原九案例；成员大表只存parquet以避免重复CSV。",
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED",
        "files": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": parent.digest(path)} for path in paths]})
    print("全序列解析固定：只解析新增已收到PDF，复用8解析；3488日全部来源覆盖，0金融。", flush=True)


def missing_node(node):
    return inputs.qualify_metadata({"id": node["id"], "published": node["published"],
        "available_at": original.publication_clock(node["published"]), "taxonomy": node["taxonomy"],
        "page_clock_verified": node["page_clock_verified"], "pdf_received": node["pdf_received"],
        "pages": 0, "rows": 0, "parse_passed": False, "title_passed": False,
        "duplicate_security_rows": None, "unparsed_text_codes": None, "codes_absent_from_text": None,
        "unknown_classification_rows": None, "classification_path": None,
        "intake_status": node["status"], "source_reused": False, "first_vintage": "NOT_CERTIFIED"})


def parse_pdf(node):
    path = ROOT / node["pdf_path"]
    folder = OUT / "parsed" / node["id"]
    if (folder / "parse_result.json").exists():
        raise RuntimeError("新原件已解析，不能重复。")
    rows, raw_rows, text_codes, state, pages = [], [], set(), {}, []
    with pdfplumber.open(path) as pdf:
        title_passed = original.compact(node["period_title"]) in original.compact(pdf.pages[0].extract_text())
        page_count = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"\n=== PDF第{page_index}页 ===\n{text}\n")
            text_codes.update(re.findall(r"(?<!\d)\d{6}(?!\d)", text))
            for table_index, values in enumerate(page.extract_tables()):
                for row_index, raw in enumerate(values):
                    origin = f"{page_index}:{table_index}:{row_index}"
                    parsed = original.parse_row(raw, node["taxonomy"], state, origin)
                    raw_rows.append({"snapshot_id": node["id"], "pdf_page": page_index,
                        "table_index": table_index, "row_index": row_index,
                        "raw_cells": json.dumps(raw, ensure_ascii=False), "parsed_security_row": parsed is not None})
                    if parsed is not None:
                        rows.append({"snapshot_id": node["id"], "published": node["published"],
                            "available_at": original.publication_clock(node["published"]),
                            "pdf_page": page_index, "pdf_sha256": node["pdf_sha256"], **parsed})
            if page_index % 40 == 0:
                print(f"{node['id']}已解析{page_index}/{page_count}页。", flush=True)
            page.close()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "all_pages.txt").write_text("".join(pages), encoding="utf-8")
    pd.DataFrame(raw_rows).to_parquet(folder / "raw_table_rows.parquet", index=False)
    pd.DataFrame(rows).to_parquet(folder / "classification.parquet", index=False)
    quality = original.quality(rows, text_codes)
    quality["parse_passed"] = bool(quality["parse_passed"] and title_passed)
    result = inputs.qualify_metadata({"id": node["id"], "published": node["published"],
        "available_at": original.publication_clock(node["published"]), "taxonomy": node["taxonomy"],
        "pdf_path": node["pdf_path"], "pdf_sha256": node["pdf_sha256"], "pages": page_count,
        "raw_table_rows": len(raw_rows), "title_passed": title_passed, "page_clock_verified": node["page_clock_verified"],
        "pdf_received": True, "classification_path": str((folder / "classification.parquet").absolute().relative_to(ROOT)),
        **quality, "source_reused": False, "first_vintage": "NOT_CERTIFIED"})
    parent.write(folder / "parse_result.json", result)
    print(f"{node['id']}解析结束：{len(rows)}行，完整表={result['complete_snapshot_passed']}，逐行来源结构={result['row_source_snapshot_eligible']}。", flush=True)
    return result


def coverage(metadata):
    observed = pd.read_parquet(cohort.SOURCEFILES["observed"], columns=["date"])
    observed["date"] = pd.to_datetime(observed.date)
    if observed.date.duplicated().any() or not observed.date.is_monotonic_increasing:
        raise ValueError("原ETF日历不是唯一升序。")
    slots = observed.assign(membership_source_date=observed.date.shift(1))
    slots["decision_at"] = slots.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    meta = pd.DataFrame([{ "snapshot_id": row["id"], "published": pd.Timestamp(row["published"]),
        "available_at": original.publication_clock(row["published"]), "snapshot_taxonomy": row["taxonomy"],
        "complete_snapshot_passed": row["complete_snapshot_passed"], "row_source_snapshot_eligible": row["row_source_snapshot_eligible"],
        "pdf_received": row["pdf_received"], "page_clock_verified": row["page_clock_verified"]} for row in metadata]).sort_values("available_at")
    slots = pd.merge_asof(slots.sort_values("decision_at"), meta, left_on="decision_at", right_on="available_at", direction="backward")
    membership = pd.read_parquet(cohort.SOURCEFILES["membership"], columns=["membership_date", "symbol"])
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    if membership.duplicated(["membership_date", "symbol"]).any():
        raise ValueError("成员同日证券重复。")
    count = membership.groupby("membership_date").size()
    slots["members"] = slots.membership_source_date.map(count).fillna(0).astype(int)
    slots["membership_view_allowed"] = slots.members.eq(300)
    accepted, all_unknown = [], []
    for row in metadata:
        if row["classification_path"] is None:
            continue
        frame = pd.read_parquet(ROOT / row["classification_path"])
        frame = inputs.qualified_rows(frame, row)
        if len(frame):
            all_unknown.append(frame.loc[~frame.row_source_known].copy())
        if row["row_source_snapshot_eligible"]:
            accepted.append(frame)
    if not accepted:
        raise ValueError("全序列无逐行结构合格原件。")
    source = pd.concat(accepted, ignore_index=True)
    members = membership.merge(slots, left_on="membership_date", right_on="membership_source_date", how="inner", validate="many_to_one")
    members = members.merge(source, on=["snapshot_id", "symbol"], how="left", validate="many_to_one", suffixes=("", "_classification"))
    members["row_source_known"] = members.row_source_known.eq(True) & members.membership_view_allowed & members.row_source_snapshot_eligible.eq(True)
    known = members.groupby("date").row_source_known.sum()
    slots["classified_members"] = slots.date.map(known).fillna(0).astype(int)
    slots["unknown_members"] = np.where(slots.membership_view_allowed, 300 - slots.classified_members, np.nan)
    slots["coverage"] = np.where(slots.membership_view_allowed, slots.classified_members / 300, np.nan)
    slots["source_age_calendar_days"] = (slots.date - slots.published).dt.days
    slots["source_age_over_one_year"] = slots.source_age_calendar_days.gt(365)
    slots["evaluation_calendar"] = slots.date.ge(pd.Timestamp("2015-01-05"))
    slots["view_state"] = np.select([~slots.membership_view_allowed, slots.snapshot_id.isna(),
        ~slots.row_source_snapshot_eligible.eq(True), slots.classified_members.lt(300)],
        ["NO_VIEW_MEMBERSHIP_SOURCE_MISSING", "NO_VIEW_NO_KNOWN_PUBLICATION", "NO_VIEW_LATEST_SOURCE_STRUCTURAL_FAILURE", "PARTIAL_MEMBER_CLASSIFICATION_UNKNOWN"],
        default="ALL_300_MEMBERS_CLASSIFIED_BY_LATEST_SOURCE")
    valid = members.loc[members.row_source_known].copy()
    group_keys = ["date", "snapshot_id", "snapshot_taxonomy", "industry_key"]
    industries = valid.groupby(group_keys, as_index=False).agg(member_count=("symbol", "size"), major_name=("major_name", "first"))
    industries["count_share_of_300"] = industries.member_count / 300
    industries["meaning"] = "成员数量占比，不是指数权重或收益贡献。"
    yearly = slots.loc[slots.evaluation_calendar].assign(year=lambda frame: frame.date.dt.year).groupby("year", as_index=False).agg(
        calendar_days=("date", "size"), membership_known_days=("membership_view_allowed", "sum"),
        all_300_known_days=("classified_members", lambda values: int(values.eq(300).sum())),
        mean_member_coverage=("coverage", "mean"), source_age_max_days=("source_age_calendar_days", "max"),
        source_age_over_one_year_days=("source_age_over_one_year", "sum"))
    yearly["all_300_known_fraction_of_all_calendar"] = yearly.all_300_known_days / yearly.calendar_days
    yearly["financial_metrics"] = "NOT_COMPUTED"
    cases = slots.loc[slots.date.isin(pd.to_datetime(intake.previous.CASE_DATES))].copy()
    table("全部3488日官方发布钟_成员覆盖_源龄与未知", slots)
    table("全部实际源成员逐行分类_未知与出处保留", members, csv=False)
    table("全部行业成员数量_非指数权重或贡献", industries, csv=False)
    table("全部年份源覆盖与完整日历分母", yearly)
    table("原九固定案例全发布序列逐行覆盖", cases)
    table("全部原件未知分类或结构不合格行", pd.concat(all_unknown, ignore_index=True), csv=False)
    evaluated = slots.loc[slots.evaluation_calendar]
    return {"slots": len(slots), "evaluation_slots": len(evaluated), "actual_member_rows": len(members),
        "known_member_rows": int(members.row_source_known.sum()), "unknown_member_rows": int((~members.row_source_known).sum()),
        "industry_count_rows": len(industries), "all_300_known_evaluation_slots": int(evaluated.classified_members.eq(300).sum()),
        "partial_known_evaluation_slots": int((evaluated.classified_members.between(1, 299)).sum()),
        "zero_known_evaluation_slots": int(evaluated.classified_members.eq(0).sum()),
        "source_age_over_one_year_evaluation_slots": int(evaluated.source_age_over_one_year.sum()),
        "max_source_age_calendar_days": int(evaluated.source_age_calendar_days.max()),
        "yearly": yearly.to_dict("records"), "case_rows": cases.to_dict("records")}


def run():
    if (OUT / "PARSER_STARTED.json").exists():
        raise RuntimeError("全序列解析已经开始，不重复处理。")
    protocol = parent.read(OUT / "parser_protocol.json")
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("全序列解析冻结文件改变：" + item["path"])
    parent.write(OUT / "PARSER_STARTED.json", {"at": parent.original.now(), "new_accounts": 0})
    received = parent.read(OUT / "summary.json")
    reused, missing, selected = [], [], []
    for node in received["nodes"]:
        if node["source_reused"]:
            row = parent.read(intake.previous.OUT / "parsed" / node["id"] / "parse_result.json")
            reused.append(inputs.qualify_metadata({**row, "page_clock_verified": node["page_clock_verified"], "pdf_received": True, "source_reused": True}))
        elif node["pdf_received"] and node["page_clock_verified"]:
            selected.append(node)
        else:
            missing.append(missing_node(node))
    with ProcessPoolExecutor(max_workers=4) as executor:
        added = list(executor.map(parse_pdf, selected))
    metadata = sorted(reused + missing + added, key=lambda row: (row["published"], row["id"]))
    table("全部36发布节点_原完整门与逐行来源结构", pd.DataFrame(metadata))
    observed = coverage(metadata)
    changed = [item["path"] for item in protocol["files"] if parent.digest(ROOT / item["path"]) != item["sha256"]]
    if changed:
        raise ValueError("一次解析中原冻结文件改变：" + ";".join(changed))
    summary = {"at": parent.original.now(), "registration": intake.REGISTRATION, "decision": intake.RESULT,
        "status": "FULL_VISIBLE_RELEASE_SEQUENCE_AND_ALL_CALENDAR_ROW_COVERAGE_COMPLETED_NOT_FINANCIAL",
        "metadata": metadata, "new_pdf_parses": len(added), "reused_pdf_parses": len(reused), "missing_nodes": len(missing),
        "total_pages_read_or_reused": sum(row["pages"] for row in metadata), "new_pdf_pages_read": sum(row["pages"] for row in added),
        "total_classification_rows": sum(row["rows"] for row in metadata),
        "complete_snapshots_passed": sum(row["complete_snapshot_passed"] for row in metadata),
        "row_source_structurally_eligible_snapshots": sum(row["row_source_snapshot_eligible"] for row in metadata),
        "native_new_gets": received["logical_gets"], "request_failures_retained": received["request_failures"],
        "necessary_tests_passed": 3, "frozen_files_unchanged": len(protocol["files"]), **observed,
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "financial_admission": "NOT_ESTABLISHED_SOURCE_COVERAGE_ONLY", "goal_achieved": False}
    parent.write(OUT / "classification_summary.json", summary)
    print(f"全来源3488日汇总完成：{summary['all_300_known_evaluation_slots']}/{summary['evaluation_slots']}评价槽300分类已知；0新金融。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全行业发布序列与ETF原日历的逐行来源覆盖。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
