"""分别保存东吴旧版分栏EPS来源修正及原已成功记录的一致性。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, compact
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_guosen_history_v2 import parquet_frame
from research.forward_eps_soochow_layout_v3 import parse

SOURCE = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"
OUT = ROOT / "reports/research/510300_forward_eps_soochow_facts_v3"


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "research/forward_eps_soochow_layout_v3.py",
             ROOT / "research/forward_eps_soochow_facts_v1.py", ROOT / "tests/test_forward_eps_soochow_layout_v3.py",
             ROOT / "research/forward_eps_soochow_layout_v2.py", ROOT / "research/forward_eps_soochow_facts_v2.py",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v2/first_250_check/result.json",
             ROOT / "tests/test_forward_eps_soochow_facts_v1.py", ROOT / "docs/510300_FORWARD_EPS_SOOCHOW_LAYOUT_CORRECTION_V3.md",
             ROOT / "research/financial_annual_components_v1.py", ROOT / "research/forward_eps_guosen_history_v1.py",
             ROOT / "research/forward_eps_guosen_history_v2.py", ROOT / "research/forward_eps_guosen_layout_v2.py",
             SOURCE / "selected_before_originals.parquet",
             ROOT / "reports/research/510300_forward_eps_soochow_first_250_layout_check_v1/selected_earliest_250.parquet",
             ROOT / "reports/research/510300_forward_eps_soochow_first_250_layout_check_v1/result.json"]
    save(OUT / "manifest.json", {"registered_at": now(), "same_original_queue": True,
         "source_correction_before_new_candidate_returns": True, "original_collection_still_in_progress": True,
         "v2_successes_preserved_exactly": True, "files": [identity(p) for p in paths]}, exclusive=True)
    print("东吴无总标题财务表修正已登记，完整队列未变。", flush=True)


def run(first_250: bool = False) -> None:
    for item in read(OUT / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "第三版来源冻结输入变化"
    output = OUT / "first_250_check" if first_250 else OUT
    if not first_250 and not (SOURCE / "result.json").exists():
        raise RuntimeError("完整原件采集尚未结束，等待原进程")
    if (output / "result.json").exists():
        raise FileExistsError("本次来源提取已经完成")
    queue = pd.read_parquet(SOURCE / "selected_before_originals.parquet")
    if first_250:
        queue = queue.head(250)
        expected = pd.read_parquet(ROOT / "reports/research/510300_forward_eps_soochow_first_250_layout_check_v1/selected_earliest_250.parquet")
        pd.testing.assert_frame_equal(queue, expected)
    records, facts, preserved = [], [], 0
    for row in queue.to_dict("records"):
        aid = row["infoCode"]
        path = SOURCE / "document_records" / f"{aid}.json"
        record = {"report_id": aid, "ts_code": row["ts_code"], "status": "NO_VIEW_ORIGINAL_UNAVAILABLE"}
        if path.exists():
            original = read(path)
            pages = read(SOURCE / "page_texts" / f"{aid}.json")["pages"]
            try:
                parsed = parse(pages, original["provider_metadata"], row, ROOT / original["source"]["raw_pdf_path"])
                for fact in parsed["facts"]:
                    assert compact(fact["source_eps_row"]) in compact(pages[fact["source_page"] - 1]), "EPS原行与原页不符"
                    fact["raw_pdf_path"] = original["source"]["raw_pdf_path"]
                    fact["pdf_sha256"] = original["source"]["pdf_sha256"]
                record.update({"status": "ANNUAL_FORECAST_EPS_EXTRACTED", "source_record": path.relative_to(ROOT).as_posix(), **parsed})
                preserved += parsed["adapter"] == "V2_SUCCESS_PRESERVED_EXACTLY"
            except ValueError as exc:
                record.update({"status": "NO_VIEW_LAYOUT_OR_BASIS_UNRESOLVED", "reason": str(exc)})
        save(output / "document_facts" / f"{aid}.json", record, exclusive=True)
        records.append(record)
        facts.extend(record.get("facts", []))
        if len(records) % 250 == 0:
            print("东吴分栏EPS提取", len(records), "/", len(queue), "成功报告", sum(x["status"] == "ANNUAL_FORECAST_EPS_EXTRACTED" for x in records), flush=True)
    frame = parquet_frame(facts)
    if len(frame):
        assert not frame.duplicated(["report_id", "target_fiscal_year"]).any()
        frame.to_parquet(output / "annual_eps_forecast_vintages.parquet", index=False)
    save(output / "document_outcomes.json", {"rows": records}, exclusive=True)
    result = {"study_id": "510300_FORWARD_EPS_SOOCHOW_FACTS_V3", "completed_at": now(),
              "status": "LEGACY_AND_COLUMN_EPS_SOURCE_EXTRACTED_WITH_GAPS", "first_250_only": first_250,
              "selected_reports": len(queue), "parsed_reports": sum(x["status"] == "ANNUAL_FORECAST_EPS_EXTRACTED" for x in records),
              "forecast_eps_facts": len(facts), "v2_successful_reports_preserved": preserved,
              "companies_with_eps": int(frame.ts_code.nunique()) if len(frame) else 0,
              "unresolved_reasons": dict(Counter(x.get("reason") for x in records if x.get("reason"))),
              "new_strategy_returns_read": False, "new_models_fit": 0, "new_accounts_generated": 0, "goal_achieved": False}
    save(output / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--first-250", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run(args.first_250)
