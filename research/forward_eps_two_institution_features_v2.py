"""先在同一机构内计算盈利信号，再按公司等权汇总两机构资料。"""
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
from research.forward_eps_monthly_policy_v1 import company_features, monthly_origins

OUT = ROOT / "reports/research/510300_forward_eps_two_institution_features_v2"
SOURCES = {
    "guosen": {
        "institution_code": "80000007", "originals": "510300_forward_eps_csi_originals_v1",
        "facts": "510300_forward_eps_csi_facts_v2",
    },
    "soochow": {
        "institution_code": "80000031", "originals": "510300_forward_eps_soochow_originals_v1",
        "facts": "510300_forward_eps_soochow_facts_v3",
    },
}
PANEL = ROOT / "reports/research/510300_forward_eps_csi_directory_v1/historical_membership.parquet"
FIELDS = ["eps_growth", "profit_revision", "reported_earnings_yield"]


def finite_mean(values) -> float:
    values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def aggregate_institutions(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """每个机构每公司仅一个最新报告信号，先公司内平均，再公司间中位数。"""
    required = {"ts_code", "institution", "report_age_days", *FIELDS}
    if not required <= set(rows.columns):
        raise ValueError("两机构汇总缺少必要字段")
    if rows.duplicated(["ts_code", "institution"]).any():
        raise ValueError("同公司机构重复会造成报告数量加权")
    if not set(rows.institution) <= set(SOURCES):
        raise ValueError("输入出现未登记机构")
    pooled = []
    for code, group in rows.groupby("ts_code", sort=True):
        record = {"ts_code": code}
        for field in FIELDS:
            record[field] = finite_mean(group[field])
        has_growth = np.isfinite(pd.to_numeric(group.eps_growth, errors="coerce"))
        record["growth_institution_count"] = int(has_growth.sum())
        record["report_age_days"] = finite_mean(group.loc[has_growth, "report_age_days"])
        pooled.append(record)
    company = pd.DataFrame(pooled)
    summary = {}
    for prefix, frame in [("pooled", company), ("soochow", rows.loc[rows.institution.eq("soochow")])]:
        for field in FIELDS:
            values = pd.to_numeric(frame[field], errors="coerce")
            values = values[np.isfinite(values)]
            summary[f"{prefix}_{field}_company_count"] = len(values)
            summary[f"{prefix}_{field}_median"] = float(values.median()) if len(values) else np.nan
        summary[f"{prefix}_source_valid"] = (
            summary[f"{prefix}_eps_growth_company_count"] >= 30
            and summary[f"{prefix}_reported_earnings_yield_company_count"] >= 30
            and summary[f"{prefix}_profit_revision_company_count"] >= 15
        )
    growth = company.loc[np.isfinite(company.eps_growth)]
    revision = company.loc[np.isfinite(company.profit_revision), "profit_revision"]
    summary["dual_institution_growth_share"] = float(growth.growth_institution_count.eq(2).mean()) if len(growth) else np.nan
    summary["pooled_report_age_mean_days"] = finite_mean(growth.report_age_days)
    summary["pooled_profit_revision_breadth"] = float((revision.gt(0).sum() - revision.lt(0).sum()) / len(revision)) if len(revision) else np.nan
    quality = [summary[x] for x in ["dual_institution_growth_share", "pooled_report_age_mean_days", "pooled_profit_revision_breadth"]]
    summary["common_sources_valid"] = bool(summary["pooled_source_valid"] and summary["soochow_source_valid"] and np.isfinite(quality).all())
    return company, summary


def load_reports(source: str) -> tuple[dict, list[Path]]:
    settings = SOURCES[source]
    originals = ROOT / "reports/research" / settings["originals"]
    facts = ROOT / "reports/research" / settings["facts"]
    if not (originals / "result.json").exists() or not (facts / "result.json").exists():
        raise RuntimeError("原件或事实阶段仍未完成：" + source)
    if source == "soochow":
        verified = read(facts / "saved_source_verification.json")
        if verified["status"] != "PASS_SAVED_SOURCE_ROWS_CLOCKS_AND_PREVIOUS_FACTS" or verified["first_250_only"]:
            raise RuntimeError("东吴完整第三版事实尚未完成保存核对")
    report_map = {}
    for row in pd.read_parquet(originals / "selected_before_originals.parquet").to_dict("records"):
        aid = row["infoCode"]
        fact_path = facts / "document_facts" / f"{aid}.json"
        parsed = read(fact_path) if fact_path.exists() else {"facts": []}
        original_path = originals / "document_records" / f"{aid}.json"
        record = read(original_path) if original_path.exists() else None
        available = parsed.get("conservative_information_date")
        if record:
            meta = record["provider_metadata"]
            if str(meta["company_code"]) != settings["institution_code"]:
                raise ValueError("原件机构与登记机构不符")
            sequence = str(meta["eitime"])
            clocks = [str(row["publishDate"])[:10], str(meta["notice_date"])[:10], sequence[:10]]
            if parsed.get("report_internal_date"):
                clocks.append(parsed["report_internal_date"])
            if available:
                clocks.append(available)
            available = max(clocks)
        else:
            available = str(row["publishDate"])[:10]
            sequence = available + " 23:59:59"
        for fact in parsed.get("facts", []):
            if fact["institution_code"] != settings["institution_code"]:
                raise ValueError("预测事实跨机构混入")
        report_map.setdefault(row["ts_code"], []).append({"report_id": aid, "information_date": available,
            "published_sequence": sequence, "facts": parsed.get("facts", [])})
    files = [originals / "selected_before_originals.parquet", originals / "result.json",
             facts / "result.json", facts / "annual_eps_forecast_vintages.parquet", facts / "document_outcomes.json"]
    if source == "soochow":
        files.append(facts / "saved_source_verification.json")
    return report_map, files


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "tests/test_forward_eps_two_institution_features_v2.py",
             ROOT / "docs/510300_FORWARD_EPS_TWO_INSTITUTION_FEATURES_V2.md", PANEL,
             ROOT / "research/forward_eps_monthly_policy_v1.py", ROOT / "research/forward_eps_guosen_history_v1.py",
             ROOT / "research/financial_annual_components_v1.py",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v3/manifest.json",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v3/first_250_check/saved_source_verification.json",
             ROOT / "research/verify_forward_eps_soochow_facts_v3.py",
             ROOT / "reports/research/510300_forward_eps_two_institution_features_v1/manifest.json",
             ROOT / "reports/research/510300_forward_eps_soochow_originals_v1/selected_before_originals.parquet",
             ROOT / "config/510300_forward_eps_csi_facts_v2_manifest.json",
             ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"]
    save(OUT / "manifest.json", {"registered_at": now(), "sources": SOURCES,
         "same_institution_revision_required": True, "source_only_amendment_of_unrun_v1": True,
         "soochow_facts_version": 3, "company_equal_weight_after_institution_mean": True,
         "minimum_growth_and_valuation_companies": 30, "minimum_revision_companies": 15,
         "source_rules_before_complete_new_facts": True, "new_strategy_returns_read": False,
         "files": [identity(p) for p in paths]}, exclusive=True)
    print("双机构公司等权盈利及信息质量汇总规则已登记，等待完整东吴原件与事实。", flush=True)


def run() -> None:
    for item in read(OUT / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "双机构来源规则改变"
    if (OUT / "result.json").exists():
        raise FileExistsError("两机构月度因子已经完成")
    maps, input_paths = {}, []
    for name in SOURCES:
        maps[name], paths = load_reports(name)
        input_paths.extend(paths)
    save(OUT / "completed_source_receipt.json", {"prepared_at": now(), "new_labels_read": False,
         "files": [identity(p) for p in input_paths]}, exclusive=True)
    dates = pd.read_parquet(ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet", columns=["date"]).date
    origins = monthly_origins(dates, "2026-08-14")
    membership = pd.read_parquet(PANEL)
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    member_map = {d: set(g.symbol) for d, g in membership.groupby("membership_date")}
    institution_rows, company_rows, monthly_rows = [], [], []
    for origin in origins:
        members = member_map[origin]
        assert len(members) == 300
        rows = []
        for code in sorted(members):
            for institution, report_map in maps.items():
                fields = company_features(report_map.get(code, []), origin)
                record = {"origin": origin, "ts_code": code, "institution": institution,
                          "report_age_days": np.nan, **fields}
                if record.get("information_date"):
                    assert pd.Timestamp(record["information_date"]) < origin
                rows.append(record)
        frame = pd.DataFrame(rows)
        pooled, summary = aggregate_institutions(frame)
        institution_rows.extend(rows)
        pooled["origin"] = origin
        company_rows.extend(pooled.to_dict("records"))
        monthly_rows.append({"origin": origin, "actual_index_members": len(members), **summary})
    monthly = pd.DataFrame(monthly_rows)
    pd.DataFrame(institution_rows).to_parquet(OUT / "institution_company_month_evidence.parquet", index=False)
    pd.DataFrame(company_rows).to_parquet(OUT / "pooled_company_month_features.parquet", index=False)
    monthly.to_parquet(OUT / "monthly_two_institution_features.parquet", index=False)
    monthly.to_csv(OUT / "双机构月度因子与覆盖.csv", index=False, encoding="utf-8-sig")
    valid = monthly.common_sources_valid
    result = {"study_id": "510300_FORWARD_EPS_TWO_INSTITUTION_FEATURES_V2", "completed_at": now(),
              "status": "COMPANY_EQUAL_WEIGHT_TWO_INSTITUTION_FEATURES_COMPLETE", "monthly_origins": len(monthly),
              "institution_company_month_rows": len(institution_rows), "pooled_company_month_rows": len(company_rows),
              "common_valid_months": int(valid.sum()), "first_common_valid_month": str(monthly.loc[valid, "origin"].min()),
              "pooled_valid_months": int(monthly.pooled_source_valid.sum()),
              "soochow_valid_months": int(monthly.soochow_source_valid.sum()),
              "no_new_return_labels": True, "new_models_fit": 0, "new_accounts_generated": 0,
              "market_consensus": False, "exact_index_eps": False, "goal_achieved": False}
    save(OUT / "result.json", result, exclusive=True)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--freeze", action="store_true")
    options.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
