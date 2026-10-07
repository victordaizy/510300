"""一次解析已保存官方PDF及原案例成员；保留全部原行和未知，不运行策略。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import pandas as pd
import pdfplumber

from research import official_industry_classification_inputs_v1 as inputs
from research import official_industry_publication_intake_v1 as intake
from research import broker_fixed_cohort_observation_v1 as cohort

ROOT, OUT = intake.ROOT, intake.OUT


def table(name, data):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path.with_suffix(".parquet"), index=False)
    data.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "parser_protocol.json").exists():
        raise RuntimeError("隔离解析层已经登记，不覆盖。")
    tests = intake.parent.read(OUT / "parser_tests_receipt.json")
    if tests["passed"] != 6 or tests["exit_code"] != 0 or tests["inputs_sha256"] != intake.parent.digest(Path(inputs.__file__)):
        raise ValueError("六必要解析/源钟测试未通过或版本改变。")
    received = intake.parent.read(OUT / "summary.json")
    paths = [Path(__file__), Path(inputs.__file__), ROOT / "tests/test_official_industry_classification_v1.py",
        OUT / "parser_tests_receipt.json", OUT / "protocol.json", OUT / "summary.json",
        cohort.SOURCEFILES["membership"], cohort.SOURCEFILES["observed"]]
    paths.extend(ROOT / node["pdf_path"] for node in received["nodes"] if node["pdf_received"])
    intake.parent.write(OUT / "parser_protocol.json", {"at": intake.parent.original.now(),
        "registration": "TECH.R217_SUPPLEMENTARY_ISOLATED_PARSER", "result": "TECH.R218",
        "scope": "ALL_RECEIVED_SELECTED_PDF_ROWS_AND_NINE_CASE_MEMBERSHIP_CLOCKS_NOT_FINANCIAL",
        "quality": "八原件全部页；正文六位证券集合=解析集合、无重复证券或未知行分类、首标题合格才parse_passed。",
        "csrc_layout": "五列合并单元格继承可见表格层级，记录源页/表/行；换门类不得沿用上个大类，不推测新行业。",
        "capco_layout": "八列逐行明确门类、大类、制造业次类，保留CAPCO2023版本。",
        "members": "九固定案例观察日；使用ETF日历前一完整源日的300成员；成员源缺失或最新已公布快照解析失败保持NO_VIEW，不对单股票从未来或旧表补分类。",
        "clock": "公布日23:59，北京时间；09-30收盘不能用当日新公布分类，下一交易日可知。",
        "partial_collection": "只覆盖预先选定案例；8快照集合不等于全部历次发布。",
        "necessary_tests": 6, "new_accounts": 0, "new_fits": 0, "new_labels": 0,
        "financial_metrics": "NOT_COMPUTED", "financial_admission": "NOT_ESTABLISHED",
        "files": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": intake.parent.digest(path)} for path in paths]})
    print("隔离解析规则固定：8收到原件全部页、9案例和6必要测试；0新账户。", flush=True)


def parse_pdf(node):
    path = ROOT / node["pdf_path"]
    rows, raw_rows, text_codes, state, pages = [], [], set(), {}, []
    with pdfplumber.open(path) as pdf:
        first_text = inputs.compact(pdf.pages[0].extract_text())
        title_passed = inputs.compact(node["period_title"]) in first_text
        page_count = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"\n=== PDF第{page_index}页 ===\n{text}\n")
            text_codes.update(re.findall(r"(?<!\d)\d{6}(?!\d)", text))
            for table_index, values in enumerate(page.extract_tables()):
                for row_index, raw in enumerate(values):
                    origin = f"{page_index}:{table_index}:{row_index}"
                    parsed = inputs.parse_row(raw, node["taxonomy"], state, origin)
                    raw_rows.append({"snapshot_id": node["id"], "pdf_page": page_index,
                        "table_index": table_index, "row_index": row_index,
                        "raw_cells": json.dumps(raw, ensure_ascii=False), "parsed_security_row": parsed is not None})
                    if parsed is not None:
                        rows.append({"snapshot_id": node["id"], "published": node["published"],
                            "available_at": inputs.publication_clock(node["published"]),
                            "pdf_page": page_index, "pdf_sha256": node["pdf_sha256"], **parsed})
            if page_index % 25 == 0:
                print(f"{node['id']}已解析{page_index}/{page_count}页。", flush=True)
    folder = OUT / "parsed" / node["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "all_pages.txt").write_text("".join(pages), encoding="utf-8")
    pd.DataFrame(raw_rows).to_parquet(folder / "raw_table_rows.parquet", index=False)
    frame = pd.DataFrame(rows)
    frame.to_parquet(folder / "classification.parquet", index=False)
    quality = inputs.quality(rows, text_codes)
    quality["parse_passed"] = bool(quality["parse_passed"] and title_passed)
    result = {"id": node["id"], "published": node["published"], "available_at": inputs.publication_clock(node["published"]),
        "taxonomy": node["taxonomy"], "pdf_path": node["pdf_path"], "pdf_sha256": node["pdf_sha256"],
        "pages": page_count, "raw_table_rows": len(raw_rows), "title_passed": title_passed,
        "classification_path": str((folder / "classification.parquet").absolute().relative_to(ROOT)),
        **quality, "first_vintage": "NOT_CERTIFIED", "financial_admission": "NOT_ESTABLISHED"}
    intake.parent.write(folder / "parse_result.json", result)
    print(f"{node['id']}完成{page_count}页/{len(frame)}证券行，源解析通过={result['parse_passed']}。", flush=True)
    return result


def run():
    if (OUT / "PARSER_STARTED.json").exists():
        raise RuntimeError("该隔离解析已经开始，不重复处理固定原件。")
    protocol = intake.parent.read(OUT / "parser_protocol.json")
    for item in protocol["files"]:
        if intake.parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("固定解析代码或原件改变：" + item["path"])
    intake.parent.write(OUT / "PARSER_STARTED.json", {"at": intake.parent.original.now(), "new_accounts": 0})
    received = intake.parent.read(OUT / "summary.json")
    selected = [node for node in received["nodes"] if node["pdf_received"] and node["page_clock_verified"]]
    with ThreadPoolExecutor(max_workers=4) as executor:
        parsed = list(executor.map(parse_pdf, selected))
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(cohort.SOURCEFILES["observed"], columns=["date"]).date))
    membership = pd.read_parquet(cohort.SOURCEFILES["membership"])
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    case_rows, member_rows, industry_rows = [], [], []
    for date in intake.CASE_DATES:
        decision = pd.Timestamp(date)
        if decision not in calendar:
            raise ValueError("预定案例不是原完整交易日。")
        index = calendar.get_loc(decision)
        source_date = calendar[index - 1]
        members = membership.loc[membership.membership_date.eq(source_date), ["symbol"]].copy()
        if members.symbol.duplicated().any():
            raise ValueError("案例前一源日成员重复，不能合并。")
        chosen = inputs.choose_snapshot(parsed, decision + pd.Timedelta(hours=15))
        snapshot = pd.read_parquet(ROOT / chosen["classification_path"]) if chosen is not None else None
        membership_known = len(members) == 300
        classified = inputs.classify_members(members, snapshot if membership_known else None)
        classified["decision_date"], classified["membership_source_date"] = decision, source_date
        member_rows.append(classified)
        known = int(classified.classification_known.sum())
        case_rows.append({"decision_date": decision, "membership_source_date": source_date,
            "snapshot_id": chosen["id"] if chosen else None, "snapshot_publication_date": chosen["published"] if chosen else None,
            "snapshot_available_at": inputs.publication_clock(chosen["published"]) if chosen else pd.NaT,
            "taxonomy": chosen["taxonomy"] if chosen else None, "members": len(members),
            "membership_view_allowed": membership_known, "classified_members": known,
            "unknown_members": len(members) - known if membership_known else None,
            "coverage": known / len(members) if membership_known else None,
            "unknown_symbols": ";".join(classified.loc[~classified.classification_known, "symbol"]),
            "clock_passed": chosen is None or inputs.publication_clock(chosen["published"]) <= (decision + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai")})
        if chosen is not None:
            for key, group in classified.loc[classified.classification_known].groupby("industry_key"):
                industry_rows.append({"decision_date": decision, "snapshot_id": chosen["id"], "taxonomy": chosen["taxonomy"],
                    "industry_key": key, "major_name": group.major_name.iloc[0], "member_count": len(group),
                    "count_share_of_300": len(group) / 300,
                    "meaning": "成员数量占比，不是沪深300指数权重或产业因果"})
    table("八快照全部解析质量与源钟", pd.DataFrame(parsed))
    table("九固定案例公布钟与300成员分类覆盖", pd.DataFrame(case_rows))
    table("九案例全部已取得成员_已知与未知保留", pd.concat(member_rows, ignore_index=True))
    table("九案例全部已知行业_成员数量非指数权重", pd.DataFrame(industry_rows))
    summary = {"at": intake.parent.original.now(), "decision": "TECH.R218", "registration": "TECH.R217",
        "status": "BOUNDED_OFFICIAL_CLASSIFICATION_SOURCE_PARSED_NOT_FULL_FINANCIAL_ADMISSION",
        "parsed": parsed, "pdfs_received": received["pdfs_received"], "parse_passed_snapshots": sum(row["parse_passed"] for row in parsed),
        "logical_gets": received["logical_gets"], "pages_read": sum(row["pages"] for row in parsed),
        "classification_rows": sum(row["rows"] for row in parsed), "case_rows": case_rows,
        "case_member_rows": sum(len(frame) for frame in member_rows), "industry_count_rows": len(industry_rows),
        "necessary_tests_passed": 6, "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "financial_metrics": "NOT_COMPUTED", "financial_admission": "NOT_ESTABLISHED_EIGHT_CASE_SNAPSHOTS_NOT_FULL_HISTORY",
        "first_vintage": "NOT_CERTIFIED_CURRENT_RETRIEVAL_OFFICIAL_FILES", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False}
    intake.parent.write(OUT / "classification_summary.json", summary)
    print(f"官方分类全部原行及9案例{summary['case_member_rows']}成员核对结束；缺源案例未知保留，0金融。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="官方行业PDF与有限案例一次隔离解析。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
