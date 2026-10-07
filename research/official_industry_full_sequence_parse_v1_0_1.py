"""仅统一复用JSON与新时间戳的存储类型，复用25新解析完成全日历汇总。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research import official_industry_full_sequence_inputs_v1 as inputs
from research import official_industry_full_sequence_intake_v1 as intake
from research import official_industry_full_sequence_parse_v1 as original

ROOT, ORIGINAL = intake.ROOT, intake.OUT
OUT = ORIGINAL / "implementation_v1_0_1"
parent = intake.parent


def normalized_metadata(frame):
    result = frame.copy()
    if "available_at" in result:
        result["available_at"] = pd.to_datetime(result.available_at, utc=True).dt.tz_convert("Asia/Shanghai").astype("datetime64[ns, Asia/Shanghai]")
    return result


def table(name, frame, csv=True):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = normalized_metadata(frame) if "available_at" in frame else frame
    saved.to_parquet(path.with_suffix(".parquet"), index=False)
    if csv:
        saved.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("时间类型修复已登记，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["passed"] != 1 or tests["exit_code"] != 0:
        raise ValueError("实际混合时间存储回归尚未通过。")
    received = parent.read(ORIGINAL / "summary.json")
    paths = [Path(__file__), ROOT / "tests/test_official_industry_full_sequence_v1_0_1.py",
        OUT / "tests_receipt.json", ORIGINAL / "PARSER_STARTED.json", ORIGINAL / "initial_parser_failure_receipt.json"]
    protocol = parent.read(ORIGINAL / "parser_protocol.json")
    paths.extend(ROOT / row["path"] for row in protocol["files"])
    for node in received["nodes"]:
        if node["pdf_received"] and not node["source_reused"]:
            folder = ORIGINAL / "parsed" / node["id"]
            paths.extend(folder / name for name in ("parse_result.json", "classification.parquet", "raw_table_rows.parquet", "all_pages.txt"))
    paths = list(dict.fromkeys(paths))
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": intake.REGISTRATION, "decision": intake.RESULT,
        "scope": "TIMESTAMP_STORAGE_TYPE_ONLY_FINISH_FULL_SOURCE_CALENDAR_FROM_SAVED_PARSES",
        "only_change": "available_at字符串/时间戳统一带北京时间时区的pandas时间类型；实际时点不变。",
        "preserved": "全部25新/8旧解析与3取得失败，行资格、完整门、公布钟、原3488日历及成员逐值保持。",
        "new_gets": 0, "new_pdf_parses": 0, "new_accounts": 0,
        "files": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": parent.digest(path)} for path in paths]})
    print("时间存储隔离修复固定：0新GET/PDF解析/账户；从全部原保存解析完成3488日覆盖。", flush=True)


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("隔离时间汇总修复已经开始，不重复运行。")
    protocol = parent.read(OUT / "protocol.json")
    for row in protocol["files"]:
        if parent.digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("登记后的原输入改变：" + row["path"])
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts": 0})
    received = parent.read(ORIGINAL / "summary.json")
    metadata, added, reused, missing = [], [], [], []
    for node in received["nodes"]:
        if node["source_reused"]:
            saved = parent.read(intake.previous.OUT / "parsed" / node["id"] / "parse_result.json")
            row = inputs.qualify_metadata({**saved, "page_clock_verified": node["page_clock_verified"], "pdf_received": True, "source_reused": True})
            reused.append(row)
        elif node["pdf_received"] and node["page_clock_verified"]:
            row = parent.read(ORIGINAL / "parsed" / node["id"] / "parse_result.json")
            added.append(row)
        else:
            row = original.missing_node(node)
            missing.append(row)
        metadata.append(row)
    metadata.sort(key=lambda row: (row["published"], row["id"]))
    table("全部36发布节点_原完整门与逐行来源结构", pd.DataFrame(metadata))
    original.OUT = OUT
    original.table = table
    observed = original.coverage(metadata)
    changed = [row["path"] for row in protocol["files"] if parent.digest(ROOT / row["path"]) != row["sha256"]]
    if changed:
        raise ValueError("隔离汇总改变原冻结文件：" + ";".join(changed))
    result = {"at": parent.original.now(), "registration": intake.REGISTRATION, "decision": intake.RESULT,
        "status": "FULL_RELEASE_SEQUENCE_ROW_SOURCE_COVERAGE_COMPLETED_WITH_STORAGE_ONLY_FIX",
        "implementation": "v1_0_1_timestamp_storage_only", "original_failure_retained": True,
        "metadata": metadata, "new_pdf_parses": len(added), "reused_pdf_parses": len(reused), "missing_nodes": len(missing),
        "total_pages_read_or_reused": sum(row["pages"] for row in metadata), "new_pdf_pages_read": sum(row["pages"] for row in added),
        "total_classification_rows": sum(row["rows"] for row in metadata),
        "complete_snapshots_passed": sum(row["complete_snapshot_passed"] for row in metadata),
        "row_source_structurally_eligible_snapshots": sum(row["row_source_snapshot_eligible"] for row in metadata),
        "native_new_gets": received["logical_gets"], "request_failures_retained": received["request_failures"],
        "necessary_tests_passed": 3, "timestamp_regression_passed": 1, "frozen_files_unchanged": len(protocol["files"]),
        **observed, "new_gets_in_implementation": 0, "new_pdf_parses_in_implementation": 0,
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "financial_admission": "NOT_ESTABLISHED_SOURCE_COVERAGE_ONLY", "goal_achieved": False}
    parent.write(OUT / "summary.json", result)
    print(f"全日历来源完成：{result['all_300_known_evaluation_slots']}/{result['evaluation_slots']}评价槽300已知，{result['unknown_member_rows']}实际成员行未知；0金融。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅修统一时间存储，从已保存全序列解析完成原覆盖汇总。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
