"""检查候选方案内累计金额的连接条件，不把未核实链直接用于交易。"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_execution_extractor_v1 as extractor

OUT = ROOT / "reports/research/510300_corporate_repurchase_chain_diagnostic_v1"
STUDY = "510300_CORPORATE_REPURCHASE_CHAIN_DIAGNOSTIC_V1"
DAILY = ROOT / "data/raw/constituents/000300_constituent_daily.parquet"


def connect(records):
    output, state = [], {}
    eligible = [r for r in records if r["status"].startswith("EXTRACTED_") and r.get("scheme_candidate_with_budget") and r.get("within_reported_plan_cap")]
    for item in sorted(eligible, key=lambda r: (pd.Timestamp(r["known_at"]),
            pd.Timestamp(r["economic_cutoff"]) if r.get("economic_cutoff") is not None else pd.Timestamp.min,
            r["document_id"])):
        row = dict(item)
        key = row["scheme_candidate_with_budget"]
        prior = state.get(key)
        row.update(prior_document_id=None, reported_increment_cents=None, increment_status="UNKNOWN_STARTING_BASELINE")
        if prior is None:
            if row["classification"] == "FIRST_EXECUTION" or row["cumulative_cents"] == 0:
                row.update(reported_increment_cents=row["cumulative_cents"],
                    increment_status="EXPLICIT_FIRST_REPORTED_EXECUTION" if row["cumulative_cents"] else "EXPLICIT_ZERO_BASELINE")
        else:
            row["prior_document_id"] = prior["document_id"]
            delta = row["cumulative_cents"] - prior["cumulative_cents"]
            old_cutoff, new_cutoff = prior.get("economic_cutoff"), row.get("economic_cutoff")
            if old_cutoff is not None and new_cutoff is not None and pd.Timestamp(new_cutoff) < pd.Timestamp(old_cutoff):
                row["increment_status"] = "REVERSED_ECONOMIC_CUTOFF_REQUIRES_REVIEW"
            elif prior["classification"] == "COMPLETION" and delta != 0:
                row["increment_status"] = "CHANGED_AFTER_COMPLETION_REQUIRES_REVIEW"
            elif delta < 0:
                row["increment_status"] = "DECREASE_OR_REVISION_REQUIRES_REVIEW"
            else:
                row["reported_increment_cents"] = delta
                precision = max(row["reporting_resolution_cents"], prior["reporting_resolution_cents"])
                row["increment_status"] = (
                    "UNCHANGED_REPORTED_CUMULATIVE" if delta == 0 else
                    "POSITIVE_WITHIN_ONE_REPORTING_UNIT" if delta <= precision else
                    "POSITIVE_REPORTED_CHANGE")
        # 当前行只基于此前已公布状态计算，不用后来的修订重写过去增量。
        state[key] = row
        row["admitted_to_trading_feature"] = False
        output.append(row)
    return output


def checks():
    base = {"symbol": "A", "status": "EXTRACTED_AMOUNT_WITH_BOARD_ANCHOR",
        "scheme_candidate_with_budget": "A_ONE", "within_reported_plan_cap": True,
        "classification": "FIRST_EXECUTION", "known_at": "2025-01-03T23:59:59+08:00",
        "economic_cutoff": "2025-01-02", "cumulative_cents": 10000,
        "reporting_resolution_cents": 1, "document_id": "1"}
    same = {**base, "classification": "PROGRESS", "known_at": "2025-01-06T23:59:59+08:00", "document_id": "2"}
    rise = {**same, "known_at": "2025-01-07T23:59:59+08:00", "cumulative_cents": 18000, "document_id": "3"}
    finished = {**rise, "known_at": "2025-01-08T23:59:59+08:00", "classification": "COMPLETION", "document_id": "4"}
    reopen = {**rise, "known_at": "2025-01-09T23:59:59+08:00", "cumulative_cents": 19000, "document_id": "5"}
    new_plan = {**reopen, "known_at": "2025-01-10T23:59:59+08:00", "scheme_candidate_with_budget": "A_TWO", "cumulative_cents": 7000, "document_id": "6"}
    rows = connect([base, same, rise, finished, reopen, new_plan])
    assert [r["reported_increment_cents"] for r in rows] == [10000, 0, 8000, 0, None, None]
    assert rows[4]["increment_status"] == "CHANGED_AFTER_COMPLETION_REQUIRES_REVIEW"
    assert rows[5]["increment_status"] == "UNKNOWN_STARTING_BASELINE"
    assert connect([base, same]) == rows[:2]
    decline = {**rise, "cumulative_cents": 9000}
    assert connect([base, decline])[-1]["increment_status"] == "DECREASE_OR_REVISION_REQUIRES_REVIEW"
    old_same_day = {**base, "classification": "PROGRESS", "economic_cutoff": "2025-01-01", "cumulative_cents": 0, "document_id": "9"}
    same_day = connect([base, old_same_day])
    assert [r["cumulative_cents"] for r in same_day] == [0, 10000]
    return {"new_schemes_not_spliced": True, "post_completion_change_flagged": True,
        "unknown_start_not_zero": True, "future_disclosure_prefix_invariant": True,
        "negative_revision_not_sold_shares": True, "same_known_time_uses_economic_date_order": True}


def run():
    tests = checks()
    path = extractor.OUT / "results/parsed_documents.json"
    records = read(path)
    for name in ["code", "results"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "purpose": "按明确董事会日期与唯一预算范围形成候选链，检查增量、完成状态、披露精度和分母资料；结果只决定后续核实工作。",
        "candidate_identity_limit": "董事会日期及预算相同仍不保证只有一个方案；需要原方案引用和A/H股份范围核实后才能纳入交易变量。",
        "increment_rule": "同候选链的当时已知累计值求差；新链未见首购则期初基数未知；完成后变化、累计减少和经济截止日倒退另列。",
        "reporting_precision": "万元等粗单位披露的相同累计值只能叫报告数未变，不能证明期间完全没有小额买入。",
        "denominator": "只核对现有成交额与市值的可用性，不加载未来收益，不以缺失市值倒推比例。",
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in [path, DAILY]},
        "code_sha256": digest(Path(__file__)), "implementation_checks": tests,
        "new_accounts": 0, "new_fits": 0}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    chains = connect(records)
    save(OUT / "results/candidate_chain_rows.json", chains, True)
    counts = pd.Series([r["increment_status"] for r in chains]).value_counts().to_dict()
    facts = [r for r in records if r["status"].startswith("EXTRACTED_")]
    issues = []
    for row in facts:
        if row.get("within_reported_plan_cap") is False:
            issues.append({"document_id": row["document_id"], "reason": "EXTRACTED_AMOUNT_EXCEEDS_UNAMBIGUOUS_PLAN_CAP"})
        if not row.get("scheme_candidate_with_budget"):
            issues.append({"document_id": row["document_id"], "reason": "NO_SINGLE_BOARD_AND_BUDGET_IDENTITY"})
    for row in chains:
        if "REQUIRES_REVIEW" in row["increment_status"]:
            issues.append({"document_id": row["document_id"], "prior_document_id": row["prior_document_id"], "reason": row["increment_status"]})
    save(OUT / "results/identity_or_amount_issues.json", issues, True)
    daily = pd.read_parquet(DAILY, columns=["date", "con_code", "amount", "price_source", "total_market_cap_cny", "market_cap_asof_date"])
    assert not daily.duplicated(["date", "con_code"]).any()
    denominators = {"rows": len(daily), "companies": daily.con_code.nunique(),
        "first_date": daily.date.min(), "last_date": daily.date.max(),
        "missing_amount_rows": int(daily.amount.isna().sum()), "negative_amount_rows": int(daily.amount.lt(0).sum()),
        "zero_amount_rows": int(daily.amount.eq(0).sum()), "missing_market_cap_rows": int(daily.total_market_cap_cny.isna().sum()),
        "price_sources": daily.price_source.value_counts().to_dict(),
        "unit_check": "现有日线构建代码将供应商amount乘1000转换为元；后续信号须取当时已知成交日，不能使用公告之后的成交额。",
        "market_cap_ratio_admitted": False, "incomplete_after_last_date": True}
    save(OUT / "results/denominator_availability.json", denominators, True)
    result = {"at": now(), "study_id": STUDY, "status": "CANDIDATE_CHAINS_DIAGNOSED_FEATURE_ADMISSION_PENDING",
        "extracted_amounts": len(facts), "candidate_chain_rows": len(chains),
        "candidate_schemes": len({r["scheme_candidate_with_budget"] for r in chains}),
        "candidate_companies": len({r["symbol"] for r in chains}), "increment_status_counts": counts,
        "identity_or_amount_issue_rows": len(issues), "reported_cumulative_unchanged_is_not_exact_zero_flow": True,
        "daily_denominator_last_date": daily.date.max(), "market_cap_all_missing": bool(daily.total_market_cap_cny.isna().all()),
        "implementation_checks": tests, "trading_feature_admitted_documents": 0,
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print(f"候选方案连接检查完成：{len(chains)}条、{result['candidate_schemes']}个候选方案；未将未核实链直接用于交易。", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print(checks())
    else:
        run()
