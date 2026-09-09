"""在原固定队列中修正最新股本字段，保留旧事实及完整来源核对。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, compact
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_guosen_history_v2 import parquet_frame
from research.forward_eps_soochow_latest_share_layout_v4 import parse

SOURCE = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"
PREVIOUS = ROOT / "reports/research/510300_forward_eps_soochow_facts_v3"
OUT = ROOT / "reports/research/510300_forward_eps_soochow_facts_v4"
DIAGNOSTIC = ROOT / "reports/research/510300_forward_eps_soochow_latest_share_field_diagnostic_v1"


def freeze():
    OUT.mkdir(exist_ok=False, parents=True)
    files = [Path(__file__), ROOT / "research/forward_eps_soochow_latest_share_layout_v4.py",
             ROOT / "tests/test_forward_eps_soochow_latest_share_layout_v4.py",
             ROOT / "docs/510300_FORWARD_EPS_SOOCHOW_LATEST_SHARE_FIELD_V4.md",
             ROOT / "research/forward_eps_soochow_layout_v3.py", ROOT / "research/forward_eps_soochow_facts_v1.py",
             ROOT / "research/forward_eps_guosen_history_v1.py", ROOT / "research/forward_eps_guosen_history_v2.py",
             ROOT / "research/financial_annual_components_v1.py", SOURCE / "selected_before_originals.parquet",
             SOURCE / "result.json", PREVIOUS / "result.json", PREVIOUS / "document_outcomes.json",
             PREVIOUS / "annual_eps_forecast_vintages.parquet",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v1/annual_eps_forecast_vintages.parquet"]
    save(OUT / "manifest.json", {"registered_at": now(), "same_fixed_queue": True,
         "v3_successful_facts_preserved_exactly": True, "new_rule_only_for_previously_unparsed_reports": True,
         "prior_account_history_already_observed": True, "source_only_new_candidate_methods": 0,
         "files": [identity(p) for p in files]}, exclusive=True)
    print("最新股本字段修正已登记，保持6734份原件队列和第三版全部成功事实。", flush=True)


def run():
    for item in read(OUT / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "来源修正登记输入改变"
    save(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest": identity(OUT / "manifest.json")}, exclusive=True)
    queue = pd.read_parquet(SOURCE / "selected_before_originals.parquet")
    old_records = {r["report_id"]: r for r in read(PREVIOUS / "document_outcomes.json")["rows"]}
    assert len(queue) == len(old_records) == 6734
    records, facts, yearly = [], [], []
    preserved = added = 0
    for row in queue.to_dict("records"):
        rid = row["infoCode"]
        previous = old_records[rid]
        if previous.get("facts"):
            record = deepcopy(previous)
            assert record["facts"] == previous["facts"]
            preserved += 1
        else:
            record = deepcopy(previous)
            source = read(SOURCE / "document_records" / (rid + ".json"))
            pages = read(SOURCE / "page_texts" / (rid + ".json"))["pages"]
            try:
                parsed = parse(pages, source["provider_metadata"], row)
                for fact in parsed["facts"]:
                    page = compact(pages[fact["source_page"] - 1])
                    assert compact(fact["source_eps_row"]) in page
                    assert compact(fact["header_raw"]) in page
                    assert fact["header"][fact["selected_column_one_based"] - 1] == [fact["target_fiscal_year"], "E"]
                    for field in ["net_profit", "pe"]:
                        if fact.get(field + "_source_raw"):
                            assert compact(fact[field + "_source_raw"]) in page
                    fact["raw_pdf_path"] = source["source"]["raw_pdf_path"]
                    fact["pdf_sha256"] = source["source"]["pdf_sha256"]
                record = {"report_id": rid, "ts_code": row["ts_code"], "status": "ANNUAL_FORECAST_EPS_EXTRACTED",
                          "previous_v3_reason": previous.get("reason"),
                          "source_record": (SOURCE / "document_records" / (rid + ".json")).relative_to(ROOT).as_posix(), **parsed}
                added += 1
            except ValueError as exc:
                record["latest_share_field_reason"] = str(exc)
        save(OUT / "document_facts" / (rid + ".json"), record, exclusive=True)
        records.append(record)
        facts.extend(record.get("facts", []))
        yearly.append({"report_id": rid, "directory_year": int(str(row["publishDate"])[:4]),
                       "ts_code": row["ts_code"], "v3_parsed": bool(previous.get("facts")),
                       "v4_parsed": bool(record.get("facts")), "v4_newly_parsed": bool(record.get("facts")) and not bool(previous.get("facts")),
                       "previous_reason": previous.get("reason"), "v4_remaining_reason": record.get("latest_share_field_reason")})
        if len(records) % 1000 == 0:
            print("最新股本字段", len(records), "/6734，旧成功保留", preserved, "新增", added, flush=True)
    assert preserved == 4490
    frame = parquet_frame(facts)
    assert not frame.duplicated(["report_id", "target_fiscal_year"]).any()
    frame.to_parquet(OUT / "annual_eps_forecast_vintages.parquet", index=False)
    save(OUT / "document_outcomes.json", {"rows": records}, exclusive=True)
    coverage = pd.DataFrame(yearly)
    coverage.to_parquet(OUT / "report_coverage_changes.parquet", index=False)
    annual = coverage.groupby("directory_year").agg(reports=("report_id", "size"), v3_parsed=("v3_parsed", "sum"),
               v4_parsed=("v4_parsed", "sum"), newly_parsed=("v4_newly_parsed", "sum")).reset_index()
    annual.to_csv(OUT / "固定原件队列逐年提取覆盖.csv", index=False, encoding="utf-8-sig")
    v1 = pd.read_parquet(ROOT / "reports/research/510300_forward_eps_soochow_facts_v1/annual_eps_forecast_vintages.parquet")
    v3 = pd.read_parquet(PREVIOUS / "annual_eps_forecast_vintages.parquet")
    key = ["report_id", "target_fiscal_year"]
    compared = ["eps_value_exact", "source_eps_row", "source_eps_cell", "source_page", "net_profit_value_exact",
                "pe_value_exact", "conservative_information_date", "raw_pdf_path", "pdf_sha256"]
    comparison = v1.merge(v3, on=key, how="left", suffixes=("_v1", "_v3"), indicator=True, validate="one_to_one")
    assert comparison["_merge"].eq("both").all()
    for field in compared:
        assert comparison[field + "_v1"].fillna("<缺失>").astype(str).equals(comparison[field + "_v3"].fillna("<缺失>").astype(str))
    result = {"study_id": "510300_FORWARD_EPS_SOOCHOW_FACTS_V4", "completed_at": now(),
              "status": "EXPLICIT_LATEST_SHARE_DILUTED_SOURCE_CORRECTED_WITH_REMAINING_GAPS",
              "selected_reports": len(queue), "parsed_reports": preserved + added,
              "v3_successful_reports_preserved": preserved, "newly_parsed_reports": added,
              "forecast_eps_facts": len(frame), "companies_with_eps": int(frame.ts_code.nunique()),
              "yearly_directory_coverage": annual.to_dict("records"),
              "remaining_reasons": dict(Counter(r.get("latest_share_field_reason") for r in records if not r.get("facts"))),
              "new_strategy_methods": 0, "new_models_fit": 0, "new_accounts_generated": 0,
              "new_downloads": 0, "goal_achieved": False}
    save(OUT / "result.json", result, exclusive=True)
    check = {"checked_at": now(), "status": "PASS_ALL_V3_FACTS_PRESERVED_NEW_RAW_ROWS_YEARS_AND_CLOCKS",
             "v1_rows_exactly_preserved_in_v3": len(v1), "v3_reports_exactly_preserved_in_v4": preserved,
             "v3_annual_rows_exactly_preserved_in_v4": len(v3), "new_reports_checked": added,
             "all_new_raw_rows_and_forecast_years_checked": True, "new_downloads": 0,
             "new_account_evaluations": 0, "security_audit_performed": False}
    save(OUT / "saved_source_verification.json", check, exclusive=True)
    save(DIAGNOSTIC / "result.json", {"completed_at": now(), "old_version_regression_suspicion_disproved": True,
         "v1_rows_checked_against_v3": len(v1), "compared_fields": compared, "differences": 0,
         "three_first_2023_report_pages_visually_checked": True,
         "visually_checked_eps_net_profit_pe_forecast_cells": 27,
         "report_ids": ["AP202301031581650009", "AP202301041581678346", "AP202301061581758608"],
         "confirmed_omitted_field": "每股收益-最新股本摊薄（元/股）",
         "source_correction_result": identity(OUT / "result.json"), "source_verification": check,
         "new_strategy_methods": 0, "new_account_evaluations": 0}, exclusive=True)
    print("最新股本字段完整提取及旧事实核对完成。", flush=True)
    print(annual.to_string(index=False), flush=True)
    print("成功报告", preserved + added, "年度预测", len(frame), "新增报告", added, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
