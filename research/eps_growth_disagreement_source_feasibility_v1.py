"""仅核对两机构公司配对的覆盖和差异变化，不读取收益。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_eps_growth_disagreement_source_feasibility_v1.json"
PROTOCOL = ROOT / "docs/510300_EPS_GROWTH_DISAGREEMENT_SOURCE_FEASIBILITY_V1.md"
OUT = ROOT / "reports/research/510300_eps_growth_disagreement_source_feasibility_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sources(config):
    return [CONFIG, PROTOCOL, Path(__file__), ROOT / config["input"], ROOT / config["old_monthly_input"],
            ROOT / "docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md",
            ROOT / "research/forward_eps_two_institution_features_v3.py"]


def compute():
    config = read(CONFIG)
    frame = pd.read_parquet(ROOT / config["input"])
    monthly = pd.read_parquet(ROOT / config["old_monthly_input"])
    if frame.duplicated(["origin", "ts_code", "institution"]).any():
        raise ValueError("旧证据出现重复公司机构业务键")
    if set(frame.institution) != set(config["institutions"]):
        raise ValueError("机构范围不同")
    fields = ["origin", "ts_code", "eps_growth", "target_fiscal_year", "information_date", "report_age_days", "report_id"]
    first = frame.loc[frame.institution.eq("guosen"), fields]
    second = frame.loc[frame.institution.eq("soochow"), fields]
    pairs = first.merge(second, on=["origin", "ts_code"], suffixes=("_guosen", "_soochow"), how="outer", validate="one_to_one", indicator=True)
    if not pairs._merge.eq("both").all():
        raise ValueError("旧两个机构并未覆盖同一公司月度宇宙")
    pairs = pairs.drop(columns="_merge")
    finite = np.isfinite(pairs.eps_growth_guosen) & np.isfinite(pairs.eps_growth_soochow)
    pairs["paired_growth_available"] = finite
    for suffix in config["institutions"]:
        available = pd.to_datetime(pairs.loc[finite, "information_date_" + suffix])
        origins = pd.to_datetime(pairs.loc[finite, "origin"])
        if not (available < origins).all():
            raise ValueError("配对中含未到可用日的报告")
        if not pairs.loc[finite, "target_fiscal_year_" + suffix].eq(origins.dt.year + 1).all():
            raise ValueError("机构间年度不一致")
        ages = (origins - available).dt.days
        if not ages.between(1, config["maximum_report_age_days"]).all():
            raise ValueError("配对含过期报告")
        if not np.array_equal(ages.to_numpy(), pairs.loc[finite, "report_age_days_" + suffix].to_numpy()):
            raise ValueError("报告年龄与日期不一致")
    pairs["absolute_growth_disagreement"] = np.nan
    pairs.loc[finite, "absolute_growth_disagreement"] = abs(pairs.loc[finite, "eps_growth_guosen"] - pairs.loc[finite, "eps_growth_soochow"])
    if not pairs.loc[finite, "absolute_growth_disagreement"].between(0, 4).all():
        raise ValueError("对称增长之差超出数学范围")
    rows = []
    for item in monthly.itertuples(index=False):
        part = pairs.loc[pairs.origin.eq(item.origin)]
        values = part.loc[part.paired_growth_available, "absolute_growth_disagreement"]
        rows.append({"origin": item.origin, "company_universe": len(part), "paired_company_count": len(values),
                     "missing_pair_count": int((~part.paired_growth_available).sum()),
                     "median_absolute_growth_disagreement": float(values.median()) if len(values) else np.nan,
                     "old_common_sources_valid": bool(item.common_sources_valid),
                     "eligible_for_next_protocol": bool(item.common_sources_valid and len(values) >= config["minimum_paired_companies"])})
    summary = pd.DataFrame(rows)
    eligible = summary.loc[summary.eligible_for_next_protocol]
    values = eligible.median_absolute_growth_disagreement
    varies = bool(len(values) > 1 and values.nunique() > 1 and values.gt(0).any())
    passed = len(values) >= config["minimum_eligible_months_for_next_protocol"] and varies
    result = {"study_id": config["study_id"],
              "status": "SOURCE_COVERAGE_AND_VARIATION_AVAILABLE_PREDICTIVE_VALUE_NOT_TESTED" if passed else "NO_VIEW_INSUFFICIENT_PAIRED_SOURCE_COVERAGE_OR_VARIATION",
              "input_institution_company_month_rows": len(frame), "company_month_universe": len(pairs),
              "available_paired_company_months": int(finite.sum()), "source_months": len(summary),
              "old_common_valid_months": int(summary.old_common_sources_valid.sum()),
              "eligible_months": len(eligible), "minimum_paired_companies": config["minimum_paired_companies"],
              "eligible_pair_count_min": int(eligible.paired_company_count.min()) if len(eligible) else None,
              "eligible_pair_count_median": float(eligible.paired_company_count.median()) if len(eligible) else None,
              "eligible_pair_count_max": int(eligible.paired_company_count.max()) if len(eligible) else None,
              "first_eligible_origin": str(eligible.origin.min().date()) if len(eligible) else None,
              "last_eligible_origin": str(eligible.origin.max().date()) if len(eligible) else None,
              "statistic_min": float(values.min()) if len(values) else None,
              "statistic_median": float(values.median()) if len(values) else None,
              "statistic_max": float(values.max()) if len(values) else None,
              "statistic_distinct_values": int(values.nunique()), "source_screen_passed": passed,
              "all_report_clocks_and_absolute_target_years_verified_for_available_pairs": True,
              "same_day_publication_or_market_consensus_claimed": False,
              "predictive_value": "NOT_COMPUTED", "new_model_fits": 0, "new_return_labels": 0,
              "new_accounts_generated": 0, "new_network_requests": 0, "old_accounts_or_factors_modified": False,
              "goal_achieved": False}
    return pairs, summary, result


def verify(receipt):
    claim = read(OUT / "claim.json")
    for item in claim["files"]:
        if identity(ROOT / item["path"]) != item:
            raise ValueError("冻结来源或统计口径改变")
    pairs, monthly, result = compute()
    saved = read(OUT / "result.json")
    if result != {k: v for k, v in saved.items() if k != "completed_at"}:
        raise ValueError("保存可行性结果不一致")
    for expected, name in [(pairs, "company_pairs.parquet"), (monthly, "monthly_source_feasibility.parquet")]:
        actual = pd.read_parquet(OUT / name)
        for table in (expected, actual):
            table["origin"] = table.origin.astype("datetime64[ns]")
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True), check_exact=True)
    record = {"status": "PASS_SAVED_PAIRING_CLOCKS_AND_SOURCE_STATISTICS_RECOMPUTED", "completed_at": now(),
              "company_month_rows": len(pairs), "monthly_rows": len(monthly), "all_values_exactly_equal": True,
              "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False}
    save(receipt, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "verify"], required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        if args.receipt is None:
            raise ValueError("需要指定复核回执")
        verify(args.receipt)
        return
    save(OUT / "claim.json", {"started_at": now(), "new_disagreement_statistics_seen_before_claim": False,
                              "old_strategy_outcomes_seen": True, "new_strategy_preregistration": False,
                              "files": [identity(p) for p in sources(read(CONFIG))]})
    try:
        pairs, monthly, result = compute()
        pairs.to_parquet(OUT / "company_pairs.parquet", index=False)
        monthly.to_parquet(OUT / "monthly_source_feasibility.parquet", index=False)
        monthly.to_csv(OUT / "两机构增长分歧的逐月来源覆盖.csv", index=False, encoding="utf-8-sig")
        save(OUT / "result.json", {"completed_at": now(), **result})
    except Exception as exc:
        save(OUT / "failure.json", {"failed_at": now(), "status": "SOURCE_FEASIBILITY_PROGRAM_FAILED", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
