"""检查有关系缺口的报告是否实际进入旧月末因子，不重跑策略。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_valuation_consistency_features_v1 import qualify_report

OUT = ROOT / "reports/research/510300_forward_eps_saved_pe_usage_diagnostic_v1"
SOURCE = ROOT / "reports/research/510300_forward_eps_monthly_policy_v2_csi"
RELATION = ROOT / "reports/research/510300_forward_eps_report_pe_identity_v1"


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "docs/510300_FORWARD_EPS_SAVED_PE_USAGE_DIAGNOSTIC_V1.md",
             ROOT / "research/forward_eps_valuation_consistency_features_v1.py",
             ROOT / "reports/research/510300_forward_eps_valuation_consistency_features_v1/manifest.json",
             RELATION / "result.json", RELATION / "annual_eps_pe_reference_identity.parquet",
             RELATION / "original_page_adjudication.json"]
    paths.extend(SOURCE / name for name in ["company_month_feature_evidence.parquet", "monthly_forward_eps_features.parquet",
                                           "signals.parquet", "training_receipts.json", "result.json"])
    save(OUT / "protocol.json", {"registered_at": now(), "previous_goal_turn": "PROGRESS_SOURCE_EVIDENCE_AND_REGISTERED_AUTOMATIC_ROUND_20",
         "source": str(SOURCE.relative_to(ROOT)), "relation_rule": "VALUATION_CONSISTENCY_FEATURES_V1_ALREADY_FROZEN",
         "old_strategy_results_and_original_relation_findings_already_observed": True,
         "new_round_19_and_20_results_observed": False,
         "scope": ["ALL_SAVED_MONTHS", "ORIGINAL_EPS_VALID_MONTHS", "ORIGINAL_E3_PREDICTION_MONTHS", "ANY_ORIGINAL_E3_TRAINING_MONTH"],
         "modify_old_or_registered_new_models": False, "new_model_fit": False, "new_account_or_labels": False,
         "files": [identity(p) for p in paths]}, exclusive=True)
    print("原月末PE使用诊断已登记，沿用已固定关系条件，不新建模型或账户。", flush=True)


def summarize(frame, company, name):
    selected = company.loc[company.origin.isin(frame.origin) & company.original_pe_finite]
    removed = selected.loc[~selected.report_relation_eligible]
    available = frame.loc[frame.original_pe_company_count.gt(0)]
    compared = available.loc[available.qualified_pe_company_count.gt(0)]
    return {"scope": name, "months": len(frame), "months_with_original_pe": len(available),
            "raw_pe_company_months": len(selected), "excluded_pe_company_months": len(removed),
            "excluded_share": len(removed) / len(selected) if len(selected) else None,
            "unique_original_reports_used": int(selected.report_id.nunique()),
            "unique_relation_excluded_reports_used": int(removed.report_id.nunique()),
            "months_with_any_exclusion": int(available.excluded_pe_company_count.gt(0).sum()),
            "months_with_median_changed": int(compared.pe_median_absolute_change.gt(1e-12).sum()),
            "months_with_median_increased": int(compared.pe_median_change.gt(1e-12).sum()),
            "months_with_median_decreased": int(compared.pe_median_change.lt(-1e-12).sum()),
            "months_losing_original_source_coverage": int((frame.original_eps_valid & ~frame.qualified_source_valid).sum()),
            "median_absolute_factor_change": float(compared.pe_median_absolute_change.median()) if len(compared) else None,
            "maximum_absolute_factor_change": float(compared.pe_median_absolute_change.max()) if len(compared) else None,
            "minimum_qualified_company_count": int(available.qualified_pe_company_count.min()) if len(available) else None,
            "excluded_relation_statuses": removed.report_relation_status.value_counts().to_dict()}


def run():
    for item in read(OUT / "protocol.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "原月末PE诊断登记输入改变"
    if (OUT / "result.json").exists():
        raise FileExistsError("本版原月末PE使用诊断已完成")
    annual = pd.read_parquet(RELATION / "annual_eps_pe_reference_identity.parquet")
    annual = annual.loc[annual.source.eq("国信完整已提取原件")].copy()
    assert not annual.duplicated(["report_id", "target_fiscal_year"]).any()
    lookup = annual.set_index(["report_id", "target_fiscal_year"])
    qualified = {}
    for aid, group in annual.groupby("report_id", sort=True):
        facts = [{"target_fiscal_year": row.target_fiscal_year, "source_eps_cell": row.eps_cell_as_reported,
                  "pe_value_exact": row.pe_exact if pd.notna(row.pe_exact) else None} for row in group.itertuples()]
        quote = group.report_reference_price_exact.iloc[0]
        qualified[aid] = qualify_report(facts, quote if pd.notna(quote) else None)
    original_company = pd.read_parquet(SOURCE / "company_month_feature_evidence.parquet")
    company = original_company.copy()
    company["original_pe_finite"] = np.isfinite(company.reported_earnings_yield)
    company["report_relation_eligible"] = False
    company["report_relation_status"] = "NO_VIEW_NO_ORIGINAL_PE"
    company["target_year_relation_status"] = "NO_VIEW_NO_ORIGINAL_PE"
    company["qualified_reported_pe_inverse"] = np.nan
    for i, row in company.loc[company.original_pe_finite].iterrows():
        year = int(row.target_fiscal_year)
        assert year == row.origin.year + 1
        fact = lookup.loc[(row.report_id, year)]
        assert fact.ts_code == row.ts_code
        assert pd.Timestamp(fact.information_date) <= pd.Timestamp(row.information_date) < row.origin
        np.testing.assert_allclose(row.reported_earnings_yield, 1 / float(fact.pe_exact), atol=1e-12, rtol=1e-12)
        relation = qualified[row.report_id]
        company.loc[i, "report_relation_eligible"] = relation["eligible"]
        company.loc[i, "report_relation_status"] = relation["status"]
        company.loc[i, "target_year_relation_status"] = fact.status
        if relation["eligible"]:
            company.loc[i, "qualified_reported_pe_inverse"] = row.reported_earnings_yield
    old_monthly = pd.read_parquet(SOURCE / "monthly_forward_eps_features.parquet")
    signals = pd.read_parquet(SOURCE / "signals.parquet")
    prediction_months = set(signals.loc[np.isfinite(signals.E3_FORWARD_EPS), "date"])
    receipts = read(SOURCE / "training_receipts.json")["rows"]
    e3 = [r for r in receipts if r["model"] == "E3_FORWARD_EPS" and r["status"] == "TRAINED_MONTHLY_MODEL"]
    assert {pd.Timestamp(r["origin"]) for r in e3} == prediction_months
    training_months = {pd.Timestamp(d) for r in e3 for d in r["training_months"]}
    monthly_rows = []
    for old in old_monthly.itertuples():
        selected = company.loc[company.origin.eq(old.origin)]
        raw = selected.loc[selected.original_pe_finite, "reported_earnings_yield"]
        clean = selected.qualified_reported_pe_inverse.dropna()
        assert len(raw) == old.reported_earnings_yield_company_count
        np.testing.assert_allclose(raw.median(), old.reported_forward_earnings_yield_median, atol=1e-12, rtol=1e-12, equal_nan=True)
        change = float(clean.median() - raw.median()) if len(raw) and len(clean) else np.nan
        monthly_rows.append({"origin": old.origin, "original_pe_company_count": len(raw),
            "qualified_pe_company_count": len(clean), "excluded_pe_company_count": len(raw) - len(clean),
            "original_pe_inverse_median": raw.median(), "qualified_pe_inverse_median": clean.median(),
            "pe_median_change": change, "pe_median_absolute_change": abs(change),
            "original_eps_valid": old.all_eps_features_valid,
            "qualified_source_valid": bool(old.all_eps_features_valid and len(clean) >= 30),
            "original_E3_prediction_month": old.origin in prediction_months,
            "used_in_any_original_E3_training": old.origin in training_months,
            "original_eps_growth_company_count": old.eps_growth_company_count,
            "original_profit_revision_company_count": old.profit_revision_company_count})
    monthly = pd.DataFrame(monthly_rows)
    company["original_E3_prediction_month"] = company.origin.isin(prediction_months)
    company["used_in_any_original_E3_training"] = company.origin.isin(training_months)
    used = company.loc[company.original_pe_finite]
    report_use = used.groupby("report_id").agg(ts_code=("ts_code", "first"),
        first_used_month=("origin", "min"), last_used_month=("origin", "max"), company_months=("origin", "size"),
        original_prediction_months=("original_E3_prediction_month", "sum"),
        any_original_training_months=("used_in_any_original_E3_training", "sum"),
        report_relation_eligible=("report_relation_eligible", "first"),
        report_relation_status=("report_relation_status", "first")).reset_index()
    summaries = [summarize(monthly, company, "全部已保存139个月末"),
                 summarize(monthly.loc[monthly.original_eps_valid], company, "原EPS覆盖合格月末"),
                 summarize(monthly.loc[monthly.original_E3_prediction_month], company, "原EPS实际形成预测月末"),
                 summarize(monthly.loc[monthly.used_in_any_original_E3_training], company, "曾进入原EPS任意训练样本的月末")]
    pd.testing.assert_frame_equal(company[original_company.columns], original_company)
    company.to_parquet(OUT / "saved_company_pe_usage.parquet", index=False)
    monthly.to_parquet(OUT / "saved_monthly_pe_usage.parquet", index=False)
    monthly.to_csv(OUT / "原月末PE实际使用与关系条件影响.csv", index=False, encoding="utf-8-sig")
    report_use.to_csv(OUT / "实际使用报告与预测训练月份.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).drop(columns="excluded_relation_statuses").to_csv(OUT / "四种范围使用汇总.csv", index=False, encoding="utf-8-sig")
    result = {"study_id": "510300_FORWARD_EPS_SAVED_PE_USAGE_DIAGNOSTIC_V1", "completed_at": now(),
        "status": "ACTUAL_SAVED_REPORT_PE_USAGE_AND_MONTHLY_FACTOR_EFFECTS_DIAGNOSED",
        "source_company_month_rows": len(company), "source_reports": len(qualified),
        "monthly_rows": len(monthly), "original_model_prediction_months": len(prediction_months),
        "source_summaries": summaries, "old_company_columns_and_monthly_pe_reproduced": True,
        "new_models_fit": 0, "new_account_evaluations": 0, "new_labels_generated": 0, "new_downloads": 0,
        "return_or_sharpe_effect_identified": False, "registered_round_19_or_20_changed": False, "goal_achieved": False}
    save(OUT / "result.json", result, exclusive=True)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
