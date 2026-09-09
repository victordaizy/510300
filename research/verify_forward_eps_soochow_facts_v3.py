"""核对已保存EPS事实、原始表格、信息日期及旧版成功记录。"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, compact, decimal
from research.forward_eps_guosen_history_v1 import identity

SOURCE = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"
FACTS = ROOT / "reports/research/510300_forward_eps_soochow_facts_v3"
PREVIOUS = ROOT / "reports/research/510300_forward_eps_soochow_facts_v2/first_250_check"


def run(first_250: bool = False) -> None:
    output = FACTS / "first_250_check" if first_250 else FACTS
    result = read(output / "result.json")
    rows = read(output / "document_outcomes.json")["rows"]
    queue = pd.read_parquet(SOURCE / "selected_before_originals.parquet")
    if first_250:
        queue = queue.head(250)
    assert [r["report_id"] for r in rows] == queue.infoCode.tolist(), "完整队列顺序与保存结果不一致"
    assert len(rows) == result["selected_reports"]
    records = {r["report_id"]: r for r in rows}
    annual = pd.read_parquet(output / "annual_eps_forecast_vintages.parquet")
    assert not annual.duplicated(["report_id", "target_fiscal_year"]).any()
    indexed = annual.set_index(["report_id", "target_fiscal_year"])
    previous = read(PREVIOUS / "document_outcomes.json")["rows"]
    preserved = 0
    for row in previous:
        if row.get("facts"):
            assert row["facts"] == records[row["report_id"]]["facts"], "第二版成功事实发生改变"
            preserved += 1
    assert preserved == 90
    fixed_check_preserved = 0
    if not first_250:
        for row in read(FACTS / "first_250_check/document_outcomes.json")["rows"]:
            current = records[row["report_id"]]
            assert row == current, "完整提取改变了固定250份检查结果"
            fixed_check_preserved += 1
    facts_checked = optional_checked = 0
    for row in rows:
        if not row.get("facts"):
            continue
        aid = row["report_id"]
        receipt = read(SOURCE / "document_records" / f"{aid}.json")
        pages = read(SOURCE / "page_texts" / f"{aid}.json")["pages"]
        meta = receipt["provider_metadata"]
        assert str(meta["company_code"]) == "80000031"
        assert row["ts_code"][:6] in compact("\n".join(pages[:2]))
        for fact in row["facts"]:
            year = fact["target_fiscal_year"]
            column = fact["selected_column_one_based"] - 1
            assert fact["header"][column] == [year, "E"]
            assert decimal(fact["source_eps_cell"]) == decimal(fact["eps_value_exact"])
            assert compact(fact["source_eps_row"]) in compact(pages[fact["source_page"] - 1])
            dates = [fact["report_internal_date"], str(fact["provider_notice_date"])[:10],
                     str(fact["provider_eitime"])[:10], str(fact["directory_publish_date"])[:10]]
            for value in dates:
                date.fromisoformat(value)
            assert fact["conservative_information_date"] == max(dates)
            assert fact["pdf_sha256"] == receipt["source"]["pdf_sha256"]
            saved = indexed.loc[(aid, year)]
            assert decimal(saved.eps_value_exact) == decimal(fact["eps_value_exact"])
            for field in ["net_profit", "pe"]:
                value = fact.get(field + "_value_exact")
                if value is None:
                    assert pd.isna(saved[field + "_value_exact"])
                    continue
                assert decimal(saved[field + "_value_exact"]) == decimal(value)
                page = fact.get(field + "_source_page", fact["source_page"])
                assert compact(fact[field + "_source_raw"]) in compact(pages[page - 1]), "可选利润或市盈率原行不符"
                optional_checked += 1
            facts_checked += 1
    assert facts_checked == len(annual) == result["forecast_eps_facts"]
    expected = {
        ("AP201701130265715737", 2017): ("0.79", "1358.99", None),
        ("AP201701200282947431", 2017): ("0.938", "1861", None),
        ("AP201701230288189551", 2017): ("0.33", "244.8", "50.34"),
        ("AP201701090255009612", 2016): ("-0.41", "-569.4", None),
        ("AP201701120262349337", 2017): ("0.93", "45386", "7.18"),
    }
    for key, values in expected.items():
        actual = indexed.loc[key]
        for field, expected_value in zip(["eps_value_exact", "net_profit_value_exact", "pe_value_exact"], values):
            if expected_value is None:
                assert pd.isna(actual[field])
            else:
                assert decimal(actual[field]) == decimal(expected_value)
    save(output / "saved_source_verification.json", {
        "checked_at": now(), "status": "PASS_SAVED_SOURCE_ROWS_CLOCKS_AND_PREVIOUS_FACTS",
        "first_250_only": first_250, "reports_checked": len(rows), "annual_eps_facts_checked": facts_checked,
        "optional_profit_and_pe_facts_checked": optional_checked, "v2_success_reports_preserved": preserved,
        "fixed_250_report_results_preserved_in_full_run": fixed_check_preserved,
        "visually_checked_legacy_example_years": len(expected),
        "verification_code": identity(Path(__file__)), "new_models_fit": 0,
        "new_accounts_generated": 0, "new_downloads": 0, "security_audit_performed": False,
    }, exclusive=True)
    print("东吴EPS保存核对通过：", len(rows), "份原件，", facts_checked, "条年度预测，旧版90份事实一致。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-250", action="store_true")
    args = parser.parse_args()
    run(args.first_250)
