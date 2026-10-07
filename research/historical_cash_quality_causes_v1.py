"""分解旧财务篮子的现金质量变化，不重新运行失败交易规则。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_cash_quality_causes_v1"
SOURCE = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_2"
COHORT = ROOT / "reports/research/510300_factor96_t11_financial_cohorts_v1_0_1"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quarter_from_dependencies(dependencies, announcement_id, role, quarter):
    selected = dependencies[dependencies.target_announcement_id.eq(announcement_id)]
    values = selected.set_index("role").value
    profit = float(values.loc[role + "_CURRENT_YTD"])
    if quarter != 1:
        profit -= float(values.loc[role + "_PREVIOUS_YTD"])
    return profit, float(values.loc[role + "_PREVIOUS_ASSETS"])


def counts(frame):
    up = frame.L04_change.gt(0)
    return {
        "rows": len(frame), "dates": int(frame.date.nunique()), "issuers": int(frame.ts_code.nunique()),
        "company_joint_positive": int(frame.company_joint_positive.sum()),
        "quality_increase": int(up.sum()),
        "quality_increase_cash_decline": int((up & frame.delta_cash.lt(0)).sum()),
        "quality_increase_profit_decline": int((up & frame.delta_profit.lt(0)).sum()),
        "quality_increase_cash_and_profit_decline": int((up & frame.delta_cash.lt(0) & frame.delta_profit.lt(0)).sum()),
        "cash_and_profit_increase": int((frame.delta_cash.gt(0) & frame.delta_profit.gt(0)).sum()),
        "ttm_cash_negative": int(frame.ttm_cashflow.lt(0).sum()),
        "ttm_profit_negative": int(frame.ttm_profit.lt(0).sum()),
        "quarter_profit_below_prior_year": int(frame.quarter_profit.lt(frame.prior_same_quarter_profit).sum()),
        "L02_above_one_quarter_profit_below_prior_year": int((frame.L02.gt(1) & frame.quarter_profit.lt(frame.prior_same_quarter_profit)).sum()),
        "dominant_positive_components": {str(k): int(v) for k, v in frame.loc[up, "largest_positive_component"].value_counts().items()},
    }


def main():
    if (OUT / "decomposition_result.json").exists():
        raise SystemExit("本轮已保存结果，不覆盖既有分解。")
    days = pd.read_parquet(COHORT / "daily_financial_cohorts.parquet")
    positive = days[days.positive_financial_measurement].copy()
    companies = pd.read_parquet(COHORT / "company_financial_cohorts.parquet")
    selected = companies[companies.date.isin(positive.date)].copy()
    facts = pd.read_parquet(SOURCE / "repaired_member_report_measurements.parquet")
    dependencies = pd.read_parquet(SOURCE / "repaired_formula_dependencies.parquet")
    for frame in [selected, facts]:
        frame["announcement_id"] = frame.announcement_id.astype(str)
    dependencies["target_announcement_id"] = dependencies.target_announcement_id.astype(str)
    original_fields = ["announcement_id", "quarter_profit", "previous_quarter_assets", "quarter_roa",
                       "prior_same_quarter_roa", "prior2_same_quarter_roa", "seasonal_mean", "seasonal_sd",
                       "ttm_profit", "ttm_cashflow", "assets", "cash_quality", "prior_ttm_profit",
                       "prior_ttm_cashflow", "prior_assets", "prior_cash_quality", "event_publication_date"]
    frame = selected.merge(facts[original_fields], on="announcement_id", how="left", validate="one_to_one")
    assert len(positive) == 23 and len(frame) == 49 and frame.known.eq(True).all()
    assert np.isfinite(frame[["ttm_profit", "ttm_cashflow", "assets", "prior_ttm_profit", "prior_ttm_cashflow", "prior_assets"]]).all().all()
    assert frame.assets.gt(0).all() and frame.prior_assets.gt(0).all()
    inverse_mean = (1 / frame.assets + 1 / frame.prior_assets) / 2
    frame["delta_cash"] = frame.ttm_cashflow - frame.prior_ttm_cashflow
    frame["delta_profit"] = frame.ttm_profit - frame.prior_ttm_profit
    frame["cash_component"] = frame.delta_cash * inverse_mean
    frame["profit_component"] = -frame.delta_profit * inverse_mean
    frame["asset_component"] = ((frame.ttm_cashflow - frame.ttm_profit) + (frame.prior_ttm_cashflow - frame.prior_ttm_profit)) / 2 * (1 / frame.assets - 1 / frame.prior_assets)
    frame["component_sum"] = frame[["cash_component", "profit_component", "asset_component"]].sum(axis=1)
    frame["identity_error"] = frame.component_sum - frame.L04_change
    assert frame.identity_error.abs().max() < 1e-12
    frame["company_joint_positive"] = frame.L02.gt(1) & frame.L04_change.ge(0) & frame.L04_industry_z.ge(0)
    frame["cash_and_profit_both_increase"] = frame.delta_cash.gt(0) & frame.delta_profit.gt(0)
    frame["largest_positive_component"] = frame[["cash_component", "profit_component", "asset_component"]].idxmax(axis=1)
    prior_rows = []
    for row in frame.itertuples():
        quarter = pd.Timestamp(row.report_period).quarter
        p1, a1 = quarter_from_dependencies(dependencies, row.announcement_id, "L02_PRIOR_YEAR", quarter)
        p2, a2 = quarter_from_dependencies(dependencies, row.announcement_id, "L02_PRIOR_TWO_YEARS", quarter)
        assert abs(p1 / a1 - row.prior_same_quarter_roa) < 1e-12
        assert abs(p2 / a2 - row.prior2_same_quarter_roa) < 1e-12
        prior_rows.append({"announcement_id": row.announcement_id, "prior_same_quarter_profit": p1,
                           "prior2_same_quarter_profit": p2, "prior_same_quarter_assets": a1,
                           "prior2_same_quarter_assets": a2})
    frame = frame.merge(pd.DataFrame(prior_rows), on="announcement_id", validate="one_to_one")
    frame["quarter_profit_yoy_change"] = frame.quarter_profit - frame.prior_same_quarter_profit
    frame["quarter_profit_roa_yoy_change"] = frame.quarter_roa - frame.prior_same_quarter_roa
    frame["date"] = pd.to_datetime(frame.date)
    frame["report_period"] = pd.to_datetime(frame.report_period)
    frame.to_parquet(OUT / "全部49条公司财务变化.parquet", index=False)
    frame.to_csv(OUT / "全部49条公司财务变化.csv", index=False, encoding="utf-8-sig")
    day_rows = []
    for day, part in frame.groupby("date"):
        row = positive[positive.date.eq(day)].iloc[0]
        day_rows.append({"date": day.strftime("%Y-%m-%d"), "company_rows": len(part),
                         "individual_joint_positive": int(part.company_joint_positive.sum()),
                         "cash_and_profit_both_increase": int(part.cash_and_profit_both_increase.sum()),
                         "companies_with_cash_decline": int(part.delta_cash.lt(0).sum()),
                         "companies_with_profit_decline": int(part.delta_profit.lt(0).sum()),
                         "L02_cohort": float(row.L02_cohort), "L04_industry_cohort": float(row.L04_industry_cohort),
                         "L04_raw_change_cohort": float(row.L04_raw_change_cohort),
                         "issuers": "|".join(part.ts_code)})
    pd.DataFrame(day_rows).to_csv(OUT / "23个披露日的经营构成.csv", index=False, encoding="utf-8-sig")
    cases = []
    candidates = frame[frame.company_joint_positive & frame.L04_change.gt(0)].sort_values("announcement_id")
    for component in ["cash_component", "profit_component", "asset_component"]:
        part = candidates[candidates[component].gt(0)].sort_values(component, ascending=False, kind="stable")
        if not len(part):
            cases.append({"selection_component": component, "case": None})
            continue
        row = part.iloc[0]
        keep = ["announcement_id", "ts_code", "date", "report_period", "L02", "L04_change", "L04_industry_z",
                "ttm_cashflow", "prior_ttm_cashflow", "ttm_profit", "prior_ttm_profit", "assets", "prior_assets",
                "cash_component", "profit_component", "asset_component", "quarter_profit", "prior_same_quarter_profit",
                "prior2_same_quarter_profit", "seasonal_sd"]
        case = {k: row[k].strftime("%Y-%m-%d") if k in ["date", "report_period"] else
                str(row[k]) if k in ["announcement_id", "ts_code"] else float(row[k]) for k in keep}
        cases.append({"selection_component": component, "case": case})
    result = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "study_id": "510300_HISTORICAL_CASH_QUALITY_CAUSES_V1",
        "status": "COMPLETED_ARITHMETIC_DECOMPOSITION_SOURCE_CAUSES_PENDING",
        "all_companies_in_positive_cohorts": counts(frame),
        "individually_positive_companies": counts(frame[frame.company_joint_positive]),
        "single_company_positive_days": int(sum(row["company_rows"] == 1 for row in day_rows)),
        "positive_days_no_individual_joint_positive": int(sum(row["individual_joint_positive"] == 0 for row in day_rows)),
        "positive_days_no_company_cash_and_profit_both_increase": int(sum(row["cash_and_profit_both_increase"] == 0 for row in day_rows)),
        "max_arithmetic_identity_error": float(frame.identity_error.abs().max()),
        "selected_cases_by_factor_components_not_returns": cases,
        "new_accounts": 0, "new_return_tests": 0, "new_fitted_parameters": 0,
        "new_forecast_cards": 0, "goal_achieved": False,
    }
    save(OUT / "decomposition_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
