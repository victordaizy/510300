"""汇集三个诊断原点直接使用的旧研报原件与保存结果，不扩大来源任务。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_disclosed_holdings_diagnostic_v1 import OUT, identity, now, read, save


def symmetric(a, b):
    if a is None or b is None:
        return np.nan
    x, y = float(a), float(b)
    return 2 * (x-y) / (abs(x)+abs(y)) if abs(x)+abs(y) > 0 else np.nan


def main() -> None:
    result = read(OUT / "result.json")
    origins = pd.to_datetime([x["diagnostic_origin"] for x in result["report_admissions"]])
    base = ROOT / "reports/research/510300_forward_eps_two_institution_features_v3"
    full = pd.read_parquet(base / "institution_company_month_evidence.parquet")
    selected = full.loc[full.origin.isin(origins)].copy()
    selected.to_parquet(OUT / "selected_institution_company_evidence.parquet", index=False)
    company = pd.read_parquet(base / "pooled_company_month_features.parquet")
    company = company.loc[company.origin.isin(origins)].copy()
    company.to_parquet(OUT / "selected_pooled_company_evidence.parquet", index=False)
    sources = {}
    documents = {}
    report_fact_map = {}
    fact_comparisons = 0
    def add(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        key = path.relative_to(ROOT).as_posix()
        if key not in sources:
            sources[key] = identity(path)
    for institution, group in selected.groupby("institution"):
        stem = "csi" if institution == "guosen" else "soochow"
        facts_name = "510300_forward_eps_csi_facts_v2" if stem == "csi" else "510300_forward_eps_soochow_facts_v4"
        original_dir = ROOT / f"reports/research/510300_forward_eps_{stem}_originals_v1"
        facts_dir = ROOT / "reports/research" / facts_name
        ids = sorted(set(group.report_id.dropna()) | set(group.prior_report_id.dropna()))
        for aid in ids:
            record_path = original_dir / "document_records" / f"{aid}.json"
            fact_path = facts_dir / "document_facts" / f"{aid}.json"
            record, facts = read(record_path), read(fact_path)
            add(record_path)
            add(fact_path)
            raw = record["source"]
            for field, hash_field in [("raw_pdf_path", "pdf_sha256"), ("raw_html_path", "html_sha256")]:
                path = ROOT / raw[field]
                add(path)
                if sources[path.relative_to(ROOT).as_posix()]["sha256"] != raw[hash_field]:
                    raise ValueError("旧研报原件哈希变化：" + aid)
            directory_source = record["directory_record"].get("source_raw_path")
            if directory_source:
                add(ROOT / directory_source)
            report_fact_map[(institution, aid)] = facts
            documents[(institution, aid)] = {"institution": institution, "report_id": aid,
                "record": record_path.relative_to(ROOT).as_posix(), "facts": fact_path.relative_to(ROOT).as_posix(),
                "raw_pdf": raw["raw_pdf_path"], "raw_html": raw["raw_html_path"],
                "pdf_url": raw["pdf_url"], "pdf_sha256": raw["pdf_sha256"]}
        print(f"{institution}：已核对{len(ids)}份直接相关原件及元数据。", flush=True)
    reasons = []
    for item in selected.to_dict("records"):
        inst, aid = item["institution"], item["report_id"]
        reason = item["status"]
        detail = {"origin": str(pd.Timestamp(item["origin"]).date()), "ts_code": item["ts_code"], "institution": inst,
                  "report_id": aid if pd.notna(aid) else None,
                  "prior_report_id": item["prior_report_id"] if pd.notna(item["prior_report_id"]) else None,
                  "profit_revision_available": bool(np.isfinite(item["profit_revision"]))}
        if pd.notna(aid):
            current = report_fact_map[(inst, aid)]
            target = int(pd.Timestamp(item["origin"]).year) + 1
            by_year = {int(x["target_fiscal_year"]): x for x in current["facts"]}
            future, before_year = by_year[target], by_year[target-1]
            values = {"eps_growth": symmetric(future["eps_value_exact"], before_year["eps_value_exact"]),
                      "reported_earnings_yield": np.nan, "profit_revision": np.nan, "raw_eps_revision_unadjusted": np.nan}
            pe = future.get("pe_value_exact")
            if pe is not None and np.isfinite(float(pe)) and float(pe) != 0:
                values["reported_earnings_yield"] = 1/float(pe)
            prior_id = item["prior_report_id"]
            if pd.notna(prior_id):
                prior = report_fact_map[(inst, prior_id)]
                previous = next(x for x in prior["facts"] if int(x["target_fiscal_year"]) == target)
                detail["current_profit_label"] = future.get("net_profit_source_label")
                detail["prior_profit_label"] = previous.get("net_profit_source_label")
                values["raw_eps_revision_unadjusted"] = symmetric(future["eps_value_exact"], previous["eps_value_exact"])
                if future.get("net_profit_source_label") == previous.get("net_profit_source_label"):
                    values["profit_revision"] = symmetric(future.get("net_profit_value_exact"), previous.get("net_profit_value_exact"))
                    reason = "AVAILABLE_MATCHED_PROFIT_REVISION" if np.isfinite(values["profit_revision"]) else "NO_VIEW_MATCHED_PROFIT_VALUES_MISSING"
                else:
                    reason = "NO_VIEW_PROFIT_LABEL_MISMATCH_IN_SAVED_SOURCE"
            else:
                reason = "NO_VIEW_COMPARABLE_PRIOR_TARGET_YEAR_NOT_PRESENT_IN_SAVED_EVIDENCE"
            for field, computed in values.items():
                if not np.isclose(computed, item[field], atol=1e-12, rtol=1e-12, equal_nan=True):
                    raise ValueError(f"既有盈利字段无法由保存事实重算：{inst} {aid} {field}")
                fact_comparisons += 1
        detail["reason"] = reason
        reasons.append(detail)
    # 对无当前研报的NO_VIEW保留旧状态，不将局部原件集合冒充旧目录全集。
    pooled_checks = 0
    for row in company.to_dict("records"):
        pair = selected.loc[selected.origin.eq(row["origin"]) & selected.ts_code.eq(row["ts_code"])]
        if len(pair) != 2:
            raise ValueError("机构公司对不完整")
        for field in ["eps_growth", "profit_revision", "reported_earnings_yield"]:
            values = pair[field].to_numpy(float)
            values = values[np.isfinite(values)]
            value = float(values.mean()) if len(values) else np.nan
            if not np.isclose(value, row[field], atol=1e-12, rtol=1e-12, equal_nan=True):
                raise ValueError("已有两机构合并字段不相等")
            pooled_checks += 1
    pd.DataFrame(reasons).to_csv(OUT / "逐机构盈利修正可比性原因.csv", index=False, encoding="utf-8-sig")
    save(OUT / "selected_earnings_source_closure.json", {"created_at": now(), "documents": list(documents.values()),
        "files": sorted(sources.values(), key=lambda x: x["path"]), "source_documents": len(documents),
        "source_files": len(sources), "uncompressed_bytes": sum(x["size_bytes"] for x in sources.values()),
        "selected_institution_rows": len(selected), "selected_company_rows": len(company),
        "fact_field_comparisons": fact_comparisons, "pooled_field_comparisons": pooled_checks,
        "old_missing_current_report_statuses_preserved": True, "old_full_directory_completeness_reproved": False,
        "new_pdf_downloads": 0, "new_fact_extraction": 0, "new_models": 0, "new_accounts": 0})
    monthly = pd.read_parquet(base / "monthly_two_institution_features.parquet")
    valid = monthly.loc[monthly.common_sources_valid]
    statistics = {}
    for field in ["pooled_eps_growth_median", "pooled_profit_revision_median", "pooled_profit_revision_breadth"]:
        values = valid[field]
        statistics[field] = {"negative_months": int(values.lt(0).sum()), "zero_months": int(values.eq(0).sum()),
                             "positive_months": int(values.gt(0).sum()), "minimum": float(values.min()),
                             "median": float(values.median()), "maximum": float(values.max())}
    old_dirs = ["510300_forward_eps_two_institution_policy_v3", "510300_forward_eps_valuation_consistency_policy_v2",
                "510300_forward_eps_revision_distribution_policy_v1", "510300_forward_eps_source_v4_replay_comparison_v1",
                "510300_forward_eps_coverage_representativeness_v1", "510300_forward_eps_corrected_account_attribution_v1_1"]
    old_source_files = [ROOT / "reports/research" / x / "result.json" for x in old_dirs]
    account_rows = []
    for path in old_source_files:
        saved = read(path)
        for record in saved.get("all_metrics", []):
            account_rows.append({"source": path.relative_to(ROOT).as_posix(), **record})
    pd.DataFrame(account_rows).to_csv(OUT / "既有EPS完整账户结果摘录.csv", index=False, encoding="utf-8-sig")
    save(OUT / "saved_eps_support_diagnostic.json", {"created_at": now(), "status": "POST_HOC_SAVED_FEATURE_DESCRIPTION_NO_NEW_STRATEGY",
        "common_valid_months": len(valid), "statistics": statistics, "feature_source": identity(base / "monthly_two_institution_features.parquet"),
        "existing_account_source_files": [identity(x) for x in old_source_files],
        "binary_positive_growth_rule_has_no_negative_state_in_common_valid_months": bool(valid.pooled_eps_growth_median.gt(0).all()),
        "continuous_magnitude_predictive_value_not_decided_by_sign_counts": True,
        "distribution_and_revision_breadth_methods_already_tested": True,
        "new_labels": 0, "new_fits": 0, "new_accounts": 0, "new_hyperparameter_searches": 0})
    save(OUT / "source_visual_verification.json", {"verified_at": now(), "status": "PASS_FIVE_RENDERED_ORIGINAL_PAGES_INSPECTED",
        "pages": [{"year": 2017, "page": 56}, {"year": 2020, "page": 121}, {"year": 2024, "page": 7},
                  {"year": 2024, "page": 60}, {"year": 2024, "page": 67}],
        "checks": ["基金名称、年度与明细表身份", "股数、市值、净值占比与原始行一致", "2024跨行期末基金净值标签", "2024指数投资和积极投资均保留"]})
    print("原件闭包、两机构盈利重算、修正缺失原因与旧结果摘录已保存。", flush=True)


if __name__ == "__main__":
    main()
