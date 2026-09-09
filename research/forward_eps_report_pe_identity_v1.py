"""按原文显示精度检查EPS、报告市盈率和参考价，保留原数值与未知口径。"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, decimal
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_report_valuation_fields_v1 import parse_front_page

OUT = ROOT / "reports/research/510300_forward_eps_report_pe_identity_v1"
SOURCES = {
    "国信完整已提取原件": "510300_forward_eps_csi_facts_v2",
    "东吴固定最早250份": "510300_forward_eps_soochow_facts_v3/first_250_check",
    "东吴三个跨年样例": "510300_forward_eps_soochow_pilot_facts_v1",
}


def rounded_interval(raw: str) -> tuple[Decimal, Decimal]:
    value = decimal(str(raw))
    if not value.is_finite():
        raise ValueError("非有限原文数值不能作舍入核对")
    unit = Decimal(1).scaleb(value.as_tuple().exponent)
    half = abs(unit) / 2
    return value - half, value + half


def compare(eps_raw: str, pe_raw: str, quote_raw: str) -> dict:
    eps, pe, quote = [decimal(str(value)) for value in [eps_raw, pe_raw, quote_raw]]
    if not all(value.is_finite() for value in [eps, pe, quote]):
        raise ValueError("核对输入有非有限值")
    if quote <= 0 or pe == 0:
        return {"status": "NO_VIEW_NONPOSITIVE_REFERENCE_OR_ZERO_PE"}
    eps_bounds = rounded_interval(eps_raw)
    pe_bounds = rounded_interval(pe_raw)
    quote_bounds = rounded_interval(quote_raw)
    products = [left * right for left in eps_bounds for right in pe_bounds]
    lower, upper = min(products), max(products)
    overlap = max(lower, quote_bounds[0]) <= min(upper, quote_bounds[1])
    return {"status": "COMPATIBLE_WITH_REPORTED_DISPLAY_PRECISION" if overlap else "REFERENCE_PRICE_OUTSIDE_EPS_PE_ROUNDING_RANGE",
        "eps_times_pe_exact": str(eps * pe), "implied_price_lower": str(lower), "implied_price_upper": str(upper),
        "reference_price_lower": str(quote_bounds[0]), "reference_price_upper": str(quote_bounds[1]),
        "point_price_difference": str(eps * pe - quote), "point_relative_difference": float(eps * pe / quote - 1),
        "display_precision_relation_only": True, "currency_and_share_basis_proven": False}


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "tests/test_forward_eps_report_pe_identity_v1.py",
             ROOT / "docs/510300_FORWARD_EPS_REPORT_PE_IDENTITY_V1.md",
             ROOT / "research/forward_eps_report_valuation_fields_v1.py",
             ROOT / "research/financial_annual_components_v1.py"]
    paths.extend(ROOT / "reports/research" / folder / file for folder in SOURCES.values()
                 for file in ["result.json", "annual_eps_forecast_vintages.parquet"])
    save(OUT / "protocol.json", {"registered_at": now(), "sources": SOURCES,
        "scope_selected_before_identity_results": True, "previous_strategy_returns_already_observed": True,
        "each_displayed_number_half_last_unit": True, "arbitrary_percent_tolerance": None,
        "relation_check_not_complete_source_admission": True, "modify_existing_strategy_inputs": False,
        "new_models_or_accounts": False, "files": [identity(p) for p in paths]}, exclusive=True)
    print("EPS与市盈率关系核对已登记，使用原文精度，没有任意百分比容差。", flush=True)


def run():
    for item in read(OUT / "protocol.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"]
    if (OUT / "result.json").exists():
        raise FileExistsError("本次原件关系检查已完成")
    rows, reports = [], []
    for source, folder in SOURCES.items():
        frame = pd.read_parquet(ROOT / "reports/research" / folder / "annual_eps_forecast_vintages.parquet")
        for aid, group in frame.groupby("report_id", sort=True):
            if source == "国信完整已提取原件":
                original_path = ROOT / "reports/research/510300_forward_eps_csi_originals_v1/page_texts" / f"{aid}.json"
                original = read(original_path)
                quote_record = parse_front_page(original["pages"][0])["quote"]
                quote, raw_quote = quote_record["value"], quote_record["raw"]
                source_path = original_path.relative_to(ROOT).as_posix()
            else:
                first = group.iloc[0]
                quote = first.report_reference_close_exact if pd.notna(first.report_reference_close_exact) else None
                raw_quote = first.report_reference_close_raw if pd.notna(first.report_reference_close_raw) else None
                source_path = f"reports/research/{folder}/document_facts/{aid}.json"
            compared = []
            for fact in group.to_dict("records"):
                pe = fact["pe_value_exact"] if pd.notna(fact["pe_value_exact"]) else None
                record = {"source": source, "report_id": aid, "ts_code": fact["ts_code"], "sec_name": fact["sec_name"],
                    "target_fiscal_year": int(fact["target_fiscal_year"]), "report_internal_date": fact["report_internal_date"],
                    "information_date": fact["conservative_information_date"], "eps_exact": fact["eps_value_exact"],
                    "eps_cell_as_reported": fact["source_eps_cell"], "pe_exact": pe,
                    "report_reference_price_exact": quote, "reference_price_source_raw": raw_quote,
                    "source_eps_row": fact["source_eps_row"], "source_pe_row": fact["pe_source_raw"],
                    "source_page": fact["source_page"], "source_quote_record": source_path,
                    "raw_pdf_path": fact["raw_pdf_path"], "pdf_sha256": fact["pdf_sha256"],
                    "eps_definition": fact["eps_definition"], "facts_changed": False}
                if quote is None or pe is None:
                    record.update({"status": "NO_VIEW_MISSING_REFERENCE_PRICE_OR_PE",
                                   "missing_reference": quote is None, "missing_pe": pe is None})
                else:
                    record.update(compare(fact["source_eps_cell"], pe, quote))
                rows.append(record)
                if "implied_price_lower" in record:
                    compared.append(record)
            common_low = max((Decimal(x["implied_price_lower"]) for x in compared), default=None)
            common_high = min((Decimal(x["implied_price_upper"]) for x in compared), default=None)
            failures = [x for x in compared if x["status"] == "REFERENCE_PRICE_OUTSIDE_EPS_PE_ROUNDING_RANGE"]
            reports.append({"source": source, "report_id": aid, "ts_code": group.ts_code.iloc[0], "sec_name": group.sec_name.iloc[0],
                "forecast_years": len(group), "comparable_years": len(compared), "outside_reference_range_years": len(failures),
                "cross_year_implied_price_common_range_exists": bool(common_low <= common_high) if common_low is not None else None,
                "cross_year_common_lower": str(common_low) if common_low is not None else None,
                "cross_year_common_upper": str(common_high) if common_high is not None else None,
                "reference_price_exact": quote, "missing_input_only": not compared,
                "maximum_absolute_point_relative_difference": max((abs(x["point_relative_difference"]) for x in compared), default=None)})
    annual = pd.DataFrame(rows)
    grouped = pd.DataFrame(reports)
    annual.to_parquet(OUT / "annual_eps_pe_reference_identity.parquet", index=False)
    grouped.to_parquet(OUT / "report_reference_identity_summary.parquet", index=False)
    annual.to_csv(OUT / "逐年度EPS市盈率参考价关系.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(OUT / "逐份报告估值关系与缺口.csv", index=False, encoding="utf-8-sig")
    summary = []
    for source, selected in annual.groupby("source", sort=True):
        doc = grouped.loc[grouped.source.eq(source)]
        summary.append({"source": source, "reports": len(doc), "annual_forecast_rows": len(selected),
            "statuses": selected.status.value_counts().to_dict(), "reports_with_any_comparable_year": int(doc.comparable_years.gt(0).sum()),
            "reports_with_outside_reference_range_year": int(doc.outside_reference_range_years.gt(0).sum()),
            "comparable_reports_with_no_common_implied_price_across_years": int(doc.cross_year_implied_price_common_range_exists.eq(False).sum())})
    result = {"study_id": "510300_FORWARD_EPS_REPORT_PE_IDENTITY_V1", "completed_at": now(),
        "status": "REPORTED_EPS_PE_REFERENCE_RELATION_CHECKED_DIAGNOSTIC_ONLY",
        "source_summaries": summary, "annual_rows_checked": len(annual), "report_rows_checked": len(grouped),
        "rounding_tolerance_from_original_display_only": True, "exact_current_month_end_valuation_established": False,
        "missing_and_basis_unknown_preserved": True, "new_models_fit": 0, "new_accounts_generated": 0,
        "new_return_labels_generated": 0, "new_downloads": 0, "existing_source_or_policy_modified": False,
        "goal_achieved": False}
    save(OUT / "result.json", result, exclusive=True)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
