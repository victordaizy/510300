"""只读重算财务披露篮子及历史参考，不读价格或创建账户。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root.resolve()
    workspace = out.parents[2]
    for filename in ["definition_freeze.json", "result_freeze.json"]:
        for row in read(out / filename)["files"]:
            path = out / row["path"]
            assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    original = out / "original_definition"
    for row in read(original / "definition_freeze.json")["files"]:
        assert digest(original / row["path"]) == row["sha256"]
    name = "factor96_t11_financial_cohorts_v1.py"
    assert digest(out / "code" / name) == digest(original / "code" / name) == digest(workspace / "research" / name)
    receipt = read(out / "inputs/delivery_receipt.json")
    assert receipt["saved_output_recomputation"]["status"] == "PASS_SAVED_V2_0_2_BOUNDARY_REPAIR_AND_UNCHANGED_MEASUREMENT"
    source_archive = workspace / "history/financial_source_v2_0_2.zip"
    assert source_archive.stat().st_size == receipt["bytes"] and digest(source_archive) == receipt["sha256"]
    prefix = "reports/research/510300_factor96_financial_parser_scope_v2_0_2/"
    source_name = "repaired_member_report_measurements.parquet"
    with zipfile.ZipFile(source_archive) as archive:
        source_freeze = json.loads(archive.read(prefix + "result_freeze.json"))
        expected = next(r for r in source_freeze["files"] if r["path"] == source_name)
        with archive.open(prefix + source_name) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]
    assert digest(out / "inputs" / source_name) == expected["sha256"]
    assert read(out / "run_started.json")["financial_source_zip_sha256"] == receipt["sha256"]
    spec = importlib.util.spec_from_file_location("frozen_financial_cohort", out / "code" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = pd.read_parquet(out / "inputs" / source_name)
    daily, companies, dependencies = module.measure_financial_cohorts(source)
    tables = [("daily_financial_cohorts.parquet", daily), ("company_financial_cohorts.parquet", companies),
              ("industry_reference_dependencies.parquet", dependencies)]
    for filename, expected_frame in tables:
        pd.testing.assert_frame_equal(pd.read_parquet(out / filename), expected_frame, check_dtype=False, check_exact=True)
    assert dependencies.source_available_date.lt(dependencies.target_date).all()
    assert dependencies.source_report_period.lt(dependencies.target_report_period).all()
    oldest = pd.DatetimeIndex(dependencies.target_date) - pd.DateOffset(years=2)
    assert (dependencies.source_available_date.to_numpy() >= oldest.to_numpy()).all()
    for row in daily.loc[daily.cohort_known].itertuples(index=False):
        members = companies.loc[companies.date.eq(row.date)]
        assert members.known.all() and len(members) == row.nonfinancial_reports
        by_industry = members.groupby("industry_l1_code")[["L02", "L04_industry_z", "L04_change"]].median().median()
        np.testing.assert_allclose(by_industry.to_numpy(), [row.L02_cohort, row.L04_industry_cohort, row.L04_raw_change_cohort], rtol=0, atol=0)
        assert row.positive_financial_measurement == (row.L02_cohort > 1 and row.L04_industry_cohort >= 0 and row.L04_raw_change_cohort >= 0)
    assert not daily.loc[~daily.cohort_known, "positive_financial_measurement"].any()
    annual = daily.assign(year=daily.date.dt.year).groupby("year").agg(disclosure_days=("date", "size"),
        known_cohort_days=("cohort_known", "sum"), positive_financial_days=("positive_financial_measurement", "sum"),
        median_new_nonfinancial_reports=("nonfinancial_reports", "median")).reset_index()
    pd.testing.assert_frame_equal(pd.read_csv(out / "yearly_cohort_measurement.csv"), annual, check_dtype=False)
    saved = read(out / "result.json")
    assert saved["disclosure_days"] == len(daily) == 775
    assert saved["known_cohort_days"] == int(daily.cohort_known.sum()) == 207
    assert saved["positive_financial_days"] == int(daily.positive_financial_measurement.sum()) == 23
    assert saved["reference_dependency_rows"] == len(dependencies) == 91411
    assert saved["new_accounts"] == saved["new_returns"] == 0 and saved["T11"] == "NOT_RUN"
    assert saved["O02"] == "NOT_COMPUTED"
    positive = daily.loc[daily.positive_financial_measurement]
    concentration = read(out / "cohort_concentration_diagnostic.json")
    assert concentration["single_company_positive_days"] == int(positive.nonfinancial_reports.eq(1).sum()) == 15
    print(json.dumps({"status": "PASS_SAVED_T11_FINANCIAL_COHORTS_AND_PAST_ONLY_REFERENCES",
        "recomputed_tables": {name: len(table) for name, table in tables}, "known_cohort_days": 207,
        "positive_financial_days": 23, "single_company_positive_days": 15,
        "unchanged_original_algorithm": True, "upstream_financial_zip_sha256": receipt["sha256"],
        "source_input_matches_verified_financial_version": True,
        "future_reference_rows": 0, "network_requests": 0, "new_accounts": 0,
        "prices_or_strategy_returns_read": False, "external_review": "NOT_PERFORMED", "goal_achieved": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
