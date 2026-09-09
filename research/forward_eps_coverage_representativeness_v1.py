"""检查原研报覆盖和预测样本代表性；历史权重仅用于回看诊断。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import save, now
from research.forward_eps_guosen_history_v1 import identity

OUT = ROOT / "reports/research/510300_forward_eps_coverage_representativeness_v1"
SOURCE = ROOT / "reports/research/510300_forward_eps_monthly_policy_v2_csi"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [
        ROOT / "reports/research/510300_forward_eps_csi_originals_v1/selected_before_originals.parquet",
        ROOT / "reports/research/510300_forward_eps_csi_facts_v2/annual_eps_forecast_vintages.parquet",
        SOURCE / "company_month_feature_evidence.parquet",
        SOURCE / "monthly_forward_eps_features.parquet",
        ROOT / "data/raw/constituents/000300_historical_weights.parquet",
        ROOT / "reports/research/510300_forward_eps_csi_directory_v1/historical_membership.parquet",
        ROOT / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/source_remediation_admission_manifest.json",
    ]
    save(OUT / "protocol.json", {
        "registered_at": now(),
        "scope": "既有研报年度覆盖、月度预测公司、历史权重回看代表性和同公司变化诊断",
        "weight_use": "仅同月末回看诊断，不作为当时可用输入，逐期版本凭证未补全",
        "new_returns_read": 0,
        "new_models": 0,
        "files": [identity(p) for p in [Path(__file__), *paths]],
    }, exclusive=True)
    queue = pd.read_parquet(paths[0])
    facts = pd.read_parquet(paths[1])
    company = pd.read_parquet(paths[2])
    monthly = pd.read_parquet(paths[3])
    weights = pd.read_parquet(paths[4])
    membership = pd.read_parquet(paths[5])
    company["origin"] = pd.to_datetime(company.origin)
    monthly["origin"] = pd.to_datetime(monthly.origin)
    weights["trade_date"] = pd.to_datetime(weights.trade_date)
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    queue["year"] = pd.to_datetime(queue.publishDate).dt.year
    queue["eps_parsed"] = queue.infoCode.isin(facts.report_id)
    annual = queue.groupby("year").agg(
        original_reports=("infoCode", "nunique"),
        original_companies=("ts_code", "nunique"),
        eps_parsed_reports=("eps_parsed", "sum"),
    ).reset_index()
    annual.to_csv(OUT / "研报年度覆盖.csv", index=False, encoding="utf-8-sig")
    rows, details, transitions = [], [], []
    previous = None
    for origin, group in company.groupby("origin", sort=True):
        row = {"origin": origin}
        current = group.set_index("ts_code")
        member_set = set(membership.loc[membership.membership_date.eq(origin), "symbol"])
        assert len(member_set) == 300
        assert set(group.ts_code) <= member_set
        row["member_count"] = len(member_set)
        row["company_rows"] = len(group)
        w = weights.loc[weights.trade_date.eq(origin)].copy()
        row["vendor_weight_snapshot_available"] = len(w) > 0
        row["version_proven_weight_input"] = False
        if len(w):
            assert len(w) == 300 and not w.con_code.duplicated().any()
            assert set(w.con_code) == member_set
            w["normalized_weight"] = w.weight / w.weight.sum()
            w = w.merge(group, left_on="con_code", right_on="ts_code", how="left", validate="one_to_one")
            w["origin"] = origin
            row["vendor_weight_sum_pct"] = float(w.weight.sum())
            for feature in ["eps_growth", "profit_revision", "reported_earnings_yield"]:
                valid = np.isfinite(pd.to_numeric(w[feature], errors="coerce"))
                cover = float(w.loc[valid, "normalized_weight"].sum())
                row[feature + "_weight_coverage_diagnostic"] = cover
                row[feature + "_count"] = int(valid.sum())
            top = w.nlargest(10, "normalized_weight")
            row["top_ten_eps_covered_count"] = int(np.isfinite(top.eps_growth).sum())
            row["top_ten_missing_eps_symbols"] = ",".join(top.loc[~np.isfinite(top.eps_growth), "con_code"])
            details.extend(w[["origin", "con_code", "weight", "normalized_weight", "eps_growth", "profit_revision", "reported_earnings_yield", "report_id", "information_date", "report_age_days"]].to_dict("records"))
        valid_age = group.loc[np.isfinite(group.eps_growth), "report_age_days"]
        row["eps_report_age_median_days"] = float(valid_age.median()) if len(valid_age) else np.nan
        row["eps_reports_over_90_days_count"] = int(valid_age.gt(90).sum())
        if previous is not None:
            prior_origin, prior = previous
            for feature in ["eps_growth", "profit_revision", "reported_earnings_yield"]:
                a = pd.to_numeric(prior[feature], errors="coerce")
                b = pd.to_numeric(current[feature], errors="coerce")
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                common = sorted(set(a.index) & set(b.index))
                if len(common):
                    change = float(b.median() - a.median())
                    common_change = float(b.loc[common].median() - a.loc[common].median())
                    transitions.append({
                        "origin": origin, "prior_origin": prior_origin, "feature": feature,
                        "current_companies": len(b), "prior_companies": len(a), "common_companies": len(common),
                        "new_company_count": len(set(b.index) - set(a.index)),
                        "lost_company_count": len(set(a.index) - set(b.index)),
                        "full_median_change": change, "common_company_median_change": common_change,
                        "universe_composition_residual": change - common_change,
                        "annual_target_roll": origin.year != prior_origin.year,
                    })
        previous = origin, current
        rows.append(row)
    diagnostic = pd.DataFrame(rows).merge(monthly[["origin", "all_eps_features_valid"]], on="origin", validate="one_to_one")
    diagnostic.to_parquet(OUT / "monthly_coverage_diagnostic.parquet", index=False)
    diagnostic.to_csv(OUT / "月度预测公司与回看权重覆盖.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_parquet(OUT / "company_weight_diagnostic.parquet", index=False)
    pd.DataFrame(transitions).to_parquet(OUT / "monthly_company_composition_diagnostic.parquet", index=False)
    valid = diagnostic.loc[diagnostic.all_eps_features_valid]
    result = {
        "study_id": "510300_FORWARD_EPS_COVERAGE_REPRESENTATIVENESS_V1",
        "completed_at": now(), "status": "COVERAGE_AND_COMPOSITION_DIAGNOSTIC_COMPLETE_WEIGHTS_NOT_PIT_ADMITTED",
        "annual_original_report_counts": dict(zip(annual.year.astype(str), annual.original_reports.astype(int))),
        "original_reports_through_2021": int(annual.loc[annual.year.le(2021), "original_reports"].sum()),
        "parsed_reports_through_2021": int(annual.loc[annual.year.le(2021), "eps_parsed_reports"].sum()),
        "monthly_origins": len(diagnostic), "valid_eps_months": len(valid),
        "valid_eps_company_count_range": [int(valid.eps_growth_count.min()), int(valid.eps_growth_count.max())],
        "valid_eps_vendor_weight_coverage_range_diagnostic": [float(valid.eps_growth_weight_coverage_diagnostic.min()), float(valid.eps_growth_weight_coverage_diagnostic.max())],
        "valid_eps_vendor_weight_coverage_median_diagnostic": float(valid.eps_growth_weight_coverage_diagnostic.median()),
        "valid_revision_vendor_weight_coverage_range_diagnostic": [float(valid.profit_revision_weight_coverage_diagnostic.min()), float(valid.profit_revision_weight_coverage_diagnostic.max())],
        "valid_month_top_ten_covered_count_range": [int(valid.top_ten_eps_covered_count.min()), int(valid.top_ten_eps_covered_count.max())],
        "weight_version_proof_repaired": False, "weighted_forecast_factor_admitted": False,
        "new_returns_read": 0, "new_models_fit": 0, "new_accounts_generated": 0, "goal_achieved": False,
    }
    save(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
