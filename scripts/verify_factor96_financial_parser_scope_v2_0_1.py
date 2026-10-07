"""只读核对V2.0.1字段并集修复、原批次身份及同公式复算。"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ACCEPTED_FETCH = {"REUSED_SAME_HASH_PDF", "DOWNLOADED_SAME_HASH_PDF"}
METRIC_PASS = "PASS_CURRENT_CELL_SCOPE_UNIT_AND_DUPLICATE_CONSISTENCY"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while part := stream.read(1024 * 1024):
            checksum.update(part)
    return path.stat().st_size, checksum.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_freezes(out):
    base = out.parent / "510300_factor96_financial_parser_scope_v2"
    counts = {}
    for folder, names in [(base, ["batch_freeze.json", "replay_freeze.json"]),
                          (out, ["replay_freeze.json", "result_freeze.json"])]:
        for name in names:
            frozen = read(folder / name)
            for item in frozen["files"]:
                assert identity(folder / item["path"]) == (item["bytes"], item["sha256"]), item["path"]
            counts[folder.name + "/" + name] = len(frozen["files"])
    manifest = read(out / "base_input_identity.json")
    names = {p.relative_to(base).as_posix() for p in base.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    assert names == {item["path"] for item in manifest["files"]}
    for item in manifest["files"]:
        assert identity(base / item["path"]) == (item["bytes"], item["sha256"]), item["path"]
    counts["unchanged_base_input_files"] = len(manifest["files"])
    assert read(base / "batch_freeze.json")["prior_batch_freeze_sha256"] == identity(base / "source_handoff/batch_freeze.json")[1]
    assert read(base / "replay_freeze.json")["batch_freeze_sha256"] == identity(base / "batch_freeze.json")[1]
    assert read(out / "replay_freeze.json")["base_batch_freeze_sha256"] == identity(base / "batch_freeze.json")[1]
    assert (base / "replay_scope_invalidation_addendum.json").exists()
    assert not (base / "source_repair_result.json").exists()
    assert not (base / "measurement_replay_result.json").exists()
    return counts


def verify_sources(out, workspace):
    targets = read(out / "batch_targets.json")
    assert len(targets) == 2112 and sum(len(t["target_metrics"]) for t in targets) == 2529
    target_hashes = {target["sha256"] for target in targets}
    assert {p.stem for p in (out / "fetch_receipts").glob("*.json")} == target_hashes
    assert {p.stem for p in (out / "fetch_attempts").glob("*.json")} == target_hashes
    matched, received, parsed_statuses = set(), [], []
    metric_count, selected_count, matched_bytes = 0, 0, 0
    for target in targets:
        checksum = target["sha256"]
        receipt = read(out / "fetch_receipts" / (checksum + ".json"))
        assert receipt["expected_sha256"] == checksum and receipt["url"] == target["url"]
        assert receipt["announcement_id"] == target["announcement_id"]
        expected_requests = int(target["handoff"] == "UNATTEMPTED_ORIGINAL_PLAN" and not target["local_reuse"])
        assert receipt["logical_http_requests"] == expected_requests
        received.append(receipt)
        if receipt["status"] not in ACCEPTED_FETCH:
            assert not (out / "parsed_documents" / (checksum + ".json")).exists()
            continue
        matched.add(checksum)
        pdf = workspace / target["pdf_relative_path"]
        assert identity(pdf) == (target["expected_bytes"], checksum) == (receipt["bytes"], receipt["sha256"])
        with pdf.open("rb") as stream:
            assert stream.read(4) == b"%PDF"
        matched_bytes += receipt["bytes"]
        parsed = read(out / "parsed_documents" / (checksum + ".json"))
        assert parsed["official_pdf_sha256"] == checksum and parsed["announcement_id"] == target["announcement_id"]
        parsed_statuses.append(parsed["status"])
        metrics = parsed["metrics"]
        assert len(metrics) == len({m["metric_id"] for m in metrics})
        if parsed["status"] != "PARSED":
            assert not metrics
        metric_count += len(set(target["target_metrics"]) & {m["metric_id"] for m in metrics})
        for metric in metrics:
            locator = json.loads(metric["source_locator"])
            assert metric["verification_status"] == METRIC_PASS
            assert parsed["decisions"][metric["metric_id"]]["status"] == METRIC_PASS
            assert metric["statement_scope"] == "CONSOLIDATED_ONLY"
            scope = "PERIOD_END" if metric["metric_id"] == "TOTAL_ASSETS_END" else "YEAR_TO_DATE"
            assert metric["value_period_scope"] == scope
            assert 1 <= metric["source_page"] <= parsed["pdf_page_count"]
            assert locator["source_page"] == metric["source_page"]
            assert 1 <= locator["scope_page"] <= metric["source_page"]
            assert 1 <= locator["unit_page"] <= metric["source_page"]
            assert metric["source_raw_value"] in locator["row_cells"]
            assert metric["source_raw_value"] == locator["raw_value"]
            token = "".join(metric["source_raw_value"].split()).replace(",", "").replace("，", "").replace("−", "-")
            amount = Decimal(token.strip("()（）"))
            if token.startswith(("(", "（")):
                amount = -abs(amount)
            value = amount * Decimal(str(metric["source_unit_multiplier"]))
            assert abs(float(value) - metric["metric_value_cny"]) <= .011
            assert value == Decimal(locator["value_cny"])
            candidates = [c for c in parsed["candidates"] if c["metric_id"] == metric["metric_id"]]
            assert len(candidates) == parsed["decisions"][metric["metric_id"]]["candidate_count"]
            assert any(c == locator for c in candidates), "选定来源必须在保存候选中"
            for candidate in candidates:
                tolerance = (Decimal(locator["resolution_cny"]) + Decimal(candidate["resolution_cny"])) / 2 + Decimal(".011")
                assert abs(value - Decimal(candidate["value_cny"])) <= tolerance
            selected_count += 1
    assert {p.stem for p in (out / "parsed_documents").glob("*.json")} == matched
    download, batch = read(out / "download_complete.json"), read(out / "batch_complete.json")
    assert download["documents"] == batch["documents"] == batch["download_receipts"] == len(targets)
    assert download["logical_http_requests"] == sum(r["logical_http_requests"] for r in received)
    assert download["status_counts"] == dict(Counter(r["status"] for r in received))
    assert download["matched_pdf_bytes"] == matched_bytes
    assert batch["parsed_documents"] == len(matched)
    assert batch["parsed_status_counts"] == dict(Counter(parsed_statuses))
    assert batch["recovered_target_fields"] == metric_count
    protocol = read(out / "batch_protocol.json")
    assert download["logical_http_requests"] <= protocol["new_logical_http_requests_maximum"]
    assert protocol["prior_logical_http_attempts"] + download["logical_http_requests"] <= 2106
    return {"document_receipts": len(received), "same_original_hash_pdfs": len(matched), "same_original_pdf_bytes": matched_bytes,
            "saved_chosen_cells_checked": selected_count, "recovered_target_fields": metric_count,
            "all_pdf_tables_independently_reparsed": False}


def verify_source_regressions(out, workspace):
    parser = load_module("frozen_repaired_financial_parser", out / "code/factor96_financial_row_parser_v2.py")
    cards = read(out / "inputs/confirmed_contradictions.json")
    targets = {t["announcement_id"]: t for t in read(out / "batch_targets.json")}
    results = {}
    for card in cards:
        aid = card["announcement_id"]
        if aid not in results:
            target = targets[aid]
            result = parser.extract_official_pdf_facts((out / "inputs/pdf" / (aid + ".pdf")).read_bytes(),
                period_type=target["period_type"], report_period=target["report_period"])
            results[aid] = {m["metric_id"]: m for m in result["metrics"]}
        metric = results[aid][card["metric_id"]]
        assert abs(metric["metric_value_cny"] - card["original_page_value_cny"]) <= .011
        assert metric["metric_value_cny"] != card["archived_value_cny"]
    target = targets["1209410909"]
    result = parser.extract_official_pdf_facts((workspace / target["pdf_relative_path"]).read_bytes(),
        period_type="FY", report_period=target["report_period"])
    values = {m["metric_id"]: m for m in result["metrics"]}
    for metric, expected in [("OPERATING_CASH_FLOW_YTD", 585185023.09), ("PARENT_NET_PROFIT_YTD", 2915244576.05),
                             ("TOTAL_ASSETS_END", 59760062879.12)]:
        assert abs(values[metric]["metric_value_cny"] - expected) <= .011
    return {"pdfs_reparsed": len(results) + 1, "original_known_wrong_fields_corrected": len(cards), "new_scope_fields_checked": 3}


def verify_replay(out, workspace):
    sys.path.insert(0, str(workspace))
    from research import factor96_financial_parser_replay_v2_0_1 as replay
    base = out.parent / "510300_factor96_financial_parser_scope_v2"
    assert identity(workspace / "research/factor96_financial_parser_replay_v2_0_1.py") == identity(out / "code/factor96_financial_parser_replay_v2_0_1.py")
    assert identity(workspace / "research/factor96_financial_row_parser_v2.py") == identity(base / "code/factor96_financial_row_parser_v2.py")
    replay.OUT, replay.BASE, replay.INPUTS = out, base, base / "measurement_inputs"
    effective, additions = replay.extend_targets_with_confirmed_cases(
        read(base / "batch_targets.json"), read(base / "inputs/confirmed_contradictions.json"))
    assert effective == read(out / "effective_targets.json")
    assert additions == read(out / "scope_extensions.json") and len(additions) == 1
    assert sum(len(t["target_metrics"]) for t in effective) == 2530
    expected_facts, expected_changes, expected_source = replay.assemble_fields()
    for name, expected in [("repaired_verified_facts.parquet", expected_facts), ("field_change_ledger.parquet", expected_changes)]:
        pd.testing.assert_frame_equal(pd.read_parquet(out / name), expected, check_dtype=False, check_exact=True)
    saved_source = read(out / "source_repair_result.json")
    clean_source = replay.load_measurement_module().clean(expected_source)
    assert {k: v for k, v in saved_source.items() if k != "at"} == {k: v for k, v in clean_source.items() if k != "at"}
    assert saved_source["seven_known_regressions_passed"]
    assert saved_source["all_confirmed_regressions_passed"]
    assert saved_source["fields_in_frozen_summary_scope"] == 2529
    assert saved_source["additional_confirmed_non_summary_fields"] == 1
    assert saved_source["unchanged_legacy_fields_outside_scope"] == 33143
    failed_keys = expected_changes.loc[expected_changes.status.eq("NO_VIEW_UNRESOLVED_SOURCE_FIELD"), ["announcement_id", "metric_id"]]
    rejected = expected_facts.merge(failed_keys, on=["announcement_id", "metric_id"], how="inner", validate="one_to_one")
    assert rejected.verified_value.isna().all() and rejected.metric_value_cny.isna().all()
    assert rejected.fact_status.eq("NO_VIEW_UNRESOLVED_SUMMARY_SOURCE_REPAIR").all()
    module = replay.load_measurement_module()
    _, events, members, industry, _ = module.validated_sources(base / "measurement_inputs")
    measured, dependencies = module.measure_events(expected_facts, events, members, industry)
    daily, panel = module.daily_coverage(events, measured, members, industry)
    tables = {"repaired_member_report_measurements.parquet": measured, "repaired_formula_dependencies.parquet": dependencies,
              "repaired_daily_member_measurements.parquet": panel, "repaired_daily_coverage.parquet": daily}
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(pd.read_parquet(out / name), expected, check_dtype=False, check_exact=True)
    target = measured.loc[measured.member_at_available]
    finite = target.loc[target.joint_known]
    annual = daily.assign(year=daily.date.dt.year).groupby("year").agg(trading_days=("date", "size"),
        joint_median=("joint_known_count", "median"), joint_min=("joint_known_count", "min"), joint_max=("joint_known_count", "max"),
        coverage_median=("joint_coverage_nonfinancial", "median")).reset_index()
    pd.testing.assert_frame_equal(pd.read_csv(out / "repaired_yearly_daily_coverage.csv"), annual,
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    output = read(out / "measurement_replay_result.json")
    assert output["target_member_report_events"] == len(target)
    assert output["joint_measurable_events_after_source_repair"] == int(target.joint_known.sum())
    assert output["L02_measurable_events"] == int(target.L02_known.sum())
    assert output["L04_measurable_events"] == int(target.L04_known.sum())
    assert output["yearly_daily_coverage"] == module.clean(annual.to_dict("records"))
    quantiles = finite[["L02", "L04_change"]].quantile([0, .01, .5, .99, 1]).to_dict()
    assert output["diagnostic_quantiles"] == json.loads(json.dumps(module.clean(quantiles)))
    joint = measured.loc[measured.joint_known]
    l02 = (joint.quarter_roa - (joint.prior_same_quarter_roa + joint.prior2_same_quarter_roa) / 2) / (
        np.abs(joint.prior_same_quarter_roa - joint.prior2_same_quarter_roa) / np.sqrt(2))
    l04 = (joint.ttm_cashflow - joint.ttm_profit) / joint.assets - (joint.prior_ttm_cashflow - joint.prior_ttm_profit) / joint.prior_assets
    assert np.allclose(l02, joint.L02, rtol=1e-12, atol=1e-12)
    assert np.allclose(l04, joint.L04_change, rtol=1e-12, atol=1e-12)
    admitted = dependencies.status.eq("VERIFIED_SAVED_ORIGINAL_FACT")
    assert (dependencies.loc[admitted, "source_available_date"] <= dependencies.loc[admitted, "target_available_date"]).all()
    assert dependencies.loc[~admitted, "value"].isna().all()
    assert not measured.loc[measured.is_financial, "joint_known"].any()
    assert output["T11"] == "NOT_RUN" and output["O02"] == "NOT_COMPUTED" and output["new_accounts"] == 0
    assert not output["goal_achieved"] and not output["returns_read"]
    return {"reconstructed_field_rows": len(expected_facts), "reconstructed_change_rows": len(expected_changes),
            "unknown_fields_with_old_value_removed": len(rejected), "recomputed_tables": {k: len(v) for k, v in tables.items()},
            "independent_algebra_checked_rows": len(joint), "future_dependency_values_used": 0}


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=Path, required=True)
    options = args.parse_args()
    out = options.root.resolve()
    workspace = out.parents[2]
    base = out.parent / "510300_factor96_financial_parser_scope_v2"
    counts = verify_freezes(out)
    sources = verify_sources(base, workspace)
    print("冻结文件、原PDF哈希和保存单元格一致，继续重建字段与同公式覆盖。", flush=True)
    regressions = verify_source_regressions(base, workspace)
    replay = verify_replay(out, workspace)
    receipt = {"status": "PASS_SAVED_V2_0_1_SCOPE_UNION_AND_UNCHANGED_MEASUREMENT", "frozen_file_counts": counts,
        "source_identity_and_saved_cells": sources, "known_regressions": regressions, "replay": replay,
        "network_requests": 0, "new_accounts": 0, "new_fits": 0, "strategy_returns_read": False,
        "all_legacy_fields_reverified": False, "historical_first_publication_established": False,
        "external_review": "NOT_PERFORMED", "goal_achieved": False,
        "interpretation": "只读核对修复版本和固定公式；不能据此认定T11有效或全档案原文均已人工核验。"}
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
