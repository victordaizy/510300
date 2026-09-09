"""给原报告市盈率添加数值关系条件，保留盈利事实和原始因子。"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_report_pe_identity_v1 import compare, rounded_interval
from research.forward_eps_report_valuation_fields_v1 import parse_front_page
from research.forward_eps_two_institution_features_v3 import SOURCES

OUT = ROOT / "reports/research/510300_forward_eps_valuation_consistency_features_v2"
PARENT = ROOT / "reports/research/510300_forward_eps_two_institution_features_v3"
QUALIFIED = "qualified_reported_earnings_yield"
PASS = "PASS_DISPLAYED_EPS_PE_AND_REFERENCE_COMMON_RANGE"


def qualify_report(facts: list[dict], quote: str | None) -> dict:
    """只检查显示精度所容许的共同价格；不反推修正数值。"""
    result = {"eligible": False, "status": "NO_VIEW_REFERENCE_PRICE_MISSING", "comparable_years": 0,
              "missing_pe_years": [], "zero_pe_years": [], "annual_relations": []}
    if quote is None:
        return result
    if Decimal(quote) <= 0:
        result["status"] = "NO_VIEW_REFERENCE_PRICE_NONPOSITIVE"
        return result
    if len({int(row["target_fiscal_year"]) for row in facts}) != len(facts):
        raise ValueError("同报告年度事实重复，不能进行关系准入")
    bounds = [rounded_interval(quote)]
    for fact in facts:
        year = int(fact["target_fiscal_year"])
        pe = fact.get("pe_value_exact")
        if pe is None or pd.isna(pe):
            result["missing_pe_years"].append(year)
            continue
        if Decimal(str(pe)) == 0:
            result["zero_pe_years"].append(year)
            continue
        relation = compare(str(fact["source_eps_cell"]), str(pe), quote)
        result["annual_relations"].append({"target_fiscal_year": year, **relation})
        bounds.append((Decimal(relation["implied_price_lower"]), Decimal(relation["implied_price_upper"])))
    result["comparable_years"] = len(result["annual_relations"])
    if result["zero_pe_years"]:
        result["status"] = "NO_VIEW_EXPLICIT_ZERO_PE_IN_REPORT"
        return result
    if result["comparable_years"] < 2:
        result["status"] = "NO_VIEW_FEWER_THAN_TWO_COMPARABLE_FORECAST_YEARS"
        return result
    lower, upper = max(x[0] for x in bounds), min(x[1] for x in bounds)
    result.update({"common_price_lower": str(lower), "common_price_upper": str(upper),
                   "eligible": lower <= upper,
                   "status": PASS if lower <= upper else "NO_VIEW_REPORT_EPS_PE_REFERENCE_INCONSISTENT"})
    return result


def aggregate_qualified(institution: pd.DataFrame, company: pd.DataFrame, monthly: pd.DataFrame):
    """原EPS与利润修正不变，仅另算通过条件的机构PE倒数。"""
    if institution.duplicated(["origin", "ts_code", "institution"]).any():
        raise ValueError("同公司机构重复会改变公司内权重")
    means = institution.groupby(["origin", "ts_code"], sort=False)[QUALIFIED].mean().reset_index()
    pooled = company.merge(means, on=["origin", "ts_code"], how="left", sort=False, validate="one_to_one")
    if len(pooled) != len(company):
        raise ValueError("条件汇总改变了原公司月份行数")
    monthly_quality = pooled.groupby("origin")[QUALIFIED].agg(["count", "median"]).reset_index().rename(
        columns={"count": "qualified_pe_company_count", "median": "pooled_qualified_reported_earnings_yield_median"})
    combined = monthly.merge(monthly_quality, on="origin", how="left", sort=False, validate="one_to_one")
    combined["common_quality_features_valid"] = (
        combined.pooled_eps_growth_company_count.ge(30)
        & combined.pooled_profit_revision_company_count.ge(15)
        & combined.qualified_pe_company_count.ge(30)
        & np.isfinite(combined[["pooled_eps_growth_median", "pooled_profit_revision_median",
                              "pooled_reported_earnings_yield_median", "pooled_qualified_reported_earnings_yield_median"]]).all(axis=1)
    )
    return pooled, combined


def freeze() -> None:
    from research.forward_eps_source_v4_replay_registration import freeze_feature, freeze_policy
    freeze_feature("pe", __file__)


def run():
    for item in read(OUT / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "PE关系条件登记发生变化"
    if (OUT / "result.json").exists():
        raise FileExistsError("本版PE关系条件因子已经完成")
    if not (PARENT / "result.json").exists():
        raise RuntimeError("完整双机构来源仍在处理，不能用部分进度替代")
    source_paths = [PARENT / name for name in ["result.json", "completed_source_receipt.json",
                    "institution_company_month_evidence.parquet", "pooled_company_month_features.parquet",
                    "monthly_two_institution_features.parquet"]]
    for source in SOURCES.values():
        folder = ROOT / "reports/research" / source["facts"]
        source_paths.extend([folder / "result.json", folder / "annual_eps_forecast_vintages.parquet"])
    save(OUT / "completed_source_receipt.json", {"prepared_at": now(), "new_returns_read": False,
         "files": [identity(p) for p in source_paths]}, exclusive=True)
    qualifications, mapping = [], {}
    for institution, settings in SOURCES.items():
        folder = ROOT / "reports/research" / settings["facts"]
        facts = pd.read_parquet(folder / "annual_eps_forecast_vintages.parquet")
        for aid, group in facts.groupby("report_id", sort=True):
            if institution == "guosen":
                quote_path = ROOT / "reports/research/510300_forward_eps_csi_originals_v1/page_texts" / f"{aid}.json"
                quote_record = parse_front_page(read(quote_path)["pages"][0])["quote"]
                quote, raw = quote_record["value"], quote_record["raw"]
            else:
                first = group.iloc[0]
                quote = first.report_reference_close_exact if pd.notna(first.report_reference_close_exact) else None
                raw = first.report_reference_close_raw if pd.notna(first.report_reference_close_raw) else None
                quote_path = folder / "document_facts" / f"{aid}.json"
            qualified = qualify_report(group.to_dict("records"), quote)
            record = {"institution": institution, "report_id": aid, "ts_code": group.ts_code.iloc[0],
                      "sec_name": group.sec_name.iloc[0], "information_date": group.conservative_information_date.iloc[0],
                      "reference_price_exact": quote, "reference_price_raw": raw,
                      "raw_pdf_path": group.raw_pdf_path.iloc[0], "pdf_sha256": group.pdf_sha256.iloc[0],
                      "quote_source_record": identity(quote_path), **qualified}
            qualifications.append(record)
            mapping[(institution, aid)] = record
    save(OUT / "report_qualifications.json", {"rows": qualifications}, exclusive=True)
    institution = pd.read_parquet(PARENT / "institution_company_month_evidence.parquet")
    institution[QUALIFIED] = np.nan
    institution["pe_relation_status"] = "NO_VIEW_NO_FINITE_LATEST_REPORT_PE"
    for i, row in institution.iterrows():
        if not np.isfinite(row.reported_earnings_yield):
            continue
        record = mapping.get((row.institution, row.report_id))
        if record is None:
            raise ValueError("已保存公司PE找不到对应原始预测报告")
        assert record["ts_code"] == row.ts_code
        assert pd.Timestamp(record["information_date"]) < row.origin
        target = int(row.target_fiscal_year)
        assert target == row.origin.year + 1
        assert any(item["target_fiscal_year"] == target for item in record["annual_relations"]) or not record["eligible"]
        institution.loc[i, "pe_relation_status"] = record["status"]
        if record["eligible"]:
            institution.loc[i, QUALIFIED] = row.reported_earnings_yield
    original_company = pd.read_parquet(PARENT / "pooled_company_month_features.parquet")
    original_monthly = pd.read_parquet(PARENT / "monthly_two_institution_features.parquet")
    pooled, monthly = aggregate_qualified(institution, original_company, original_monthly)
    pd.testing.assert_frame_equal(pooled[original_company.columns], original_company)
    pd.testing.assert_frame_equal(monthly[original_monthly.columns], original_monthly)
    institution.to_parquet(OUT / "institution_company_month_evidence.parquet", index=False)
    pooled.to_parquet(OUT / "pooled_company_month_features.parquet", index=False)
    monthly.to_parquet(OUT / "monthly_two_institution_features.parquet", index=False)
    monthly.to_csv(OUT / "每月原PE与关系相容PE对照.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ["annual_relations", "quote_source_record"]}
                  for r in qualifications]).to_csv(OUT / "逐份报告PE关系条件.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for source in SOURCES:
        selected = [x for x in qualifications if x["institution"] == source]
        summaries.append({"institution": source, "reports": len(selected), "eligible": sum(x["eligible"] for x in selected),
                          "statuses": pd.Series([x["status"] for x in selected]).value_counts().to_dict()})
    valid = monthly.common_quality_features_valid
    save(OUT / "result.json", {"study_id": "510300_FORWARD_EPS_VALUATION_CONSISTENCY_FEATURES_V2", "completed_at": now(),
         "status": "REPORTED_PE_RELATION_CONDITIONED_FEATURES_COMPLETE", "source_summaries": summaries,
         "institution_company_month_rows": len(institution), "monthly_origins": len(monthly),
         "common_valid_months": int(valid.sum()), "first_common_valid_month": str(monthly.loc[valid, "origin"].min()),
         "old_monthly_and_company_feature_columns_unchanged": True,
         "exact_month_end_forward_earnings_yield": False, "currency_and_share_basis_fully_proven": False,
         "new_models_fit": 0, "new_accounts_generated": 0, "new_labels_read": False, "goal_achieved": False}, exclusive=True)
    print("原PE、关系相容PE和共同月份已保存；原EPS及利润修正逐项保持。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--freeze", action="store_true")
    options.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
