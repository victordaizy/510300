"""补入被摘要范围漏掉的已确认错误字段；原解析器与经济公式保持不变。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.factor96_financial_parser_batch_v2 import OUT as BASE, ROOT, read
from research.factor96_earnings_cashflow_measurement_v1 import digest, save, now


OUT = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_1"
INPUTS = BASE / "measurement_inputs"
PRIOR = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"


def load_measurement_module():
    spec = importlib.util.spec_from_file_location("unchanged_earnings_measurement", OUT / "code/frozen_measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extend_targets_with_confirmed_cases(targets, cases):
    """取冻结字段与已确认错误的并集，不增加文档或硬编码修正金额。"""
    output = deepcopy(targets)
    documents = {str(t["announcement_id"]): t for t in output}
    additions = []
    for case in cases:
        aid, metric = str(case["announcement_id"]), case["metric_id"]
        assert aid in documents, "已确认错误的原文不在已完成批次中，不能自动新增请求"
        target = documents[aid]
        assert target["sha256"] == case["pdf_sha256"], "已确认错误与批次原文哈希不符"
        if metric not in target["target_metrics"]:
            target["target_metrics"].append(metric)
            additions.append({"announcement_id": aid, "metric_id": metric,
                "pdf_sha256": target["sha256"], "reason": "PREVIOUSLY_CONFIRMED_ERROR_OUTSIDE_SUMMARY_PATH"})
    return output, additions


def prepare_replay():
    assert not OUT.exists() and (BASE / "batch_complete.json").exists()
    assert (BASE / "replay_scope_invalidation_addendum.json").exists()
    assert not (BASE / "source_repair_result.json").exists()
    original_targets = read(BASE / "batch_targets.json")
    cases = read(BASE / "inputs/confirmed_contradictions.json")
    targets, additions = extend_targets_with_confirmed_cases(original_targets, cases)
    assert len(targets) == 2112 and sum(len(t["target_metrics"]) for t in targets) == 2530
    assert len(additions) == 1
    assert additions[0]["announcement_id"] == "1214959878" and additions[0]["metric_id"] == "TOTAL_ASSETS_END"
    (OUT / "code").mkdir(parents=True)
    (OUT / "source_evidence").mkdir()
    for path in [Path(__file__), ROOT / "tests/test_factor96_financial_parser_replay_v2_0_1.py"]:
        shutil.copy2(path, OUT / "code" / path.name)
    shutil.copy2(BASE / "code/frozen_measurement.py", OUT / "code/frozen_measurement.py")
    for name in ["pipeline_replay.log", "pipeline_failed.json"]:
        shutil.copy2(ROOT / "data/audit/factor96_financial_scope_v2_controller_03" / name, OUT / "source_evidence" / name)
    save(OUT / "effective_targets.json", targets)
    save(OUT / "scope_extensions.json", additions)
    save(OUT / "replay_protocol.json", {"at": now(), "study_id": "510300_FACTOR96_FINANCIAL_SCOPE_EXTENSION_V2_0_1",
        "reason": "原V2重算因漏列一项此前已确认的非摘要路径错误而停止，未形成独立字段版本或财务测量结果。",
        "scope": "原2529个摘要字段与全部7项已确认错误取并集，新增北方华创2022Q3总资产1字段，共2530字段；仍是原2112文档、2085已解析PDF。",
        "new_field_source": "使用已保存的同哈希PDF的V2解析金额40895706832.37元；金额仍由通用解析结果提供，不能用反例卡常量替换数据。",
        "unchanged": "解析代码、全部下载记录、2085份解析结果、单季拆分、真实TTM、两年前同季窗口、披露日期代理及200日报告年龄全部保持。",
        "other_fields": "范围外33143字段明确沿用旧档案，未全部重新核实。",
        "admission": "全部7个旧反例及白云山合并现金流反例都必须在最终字段版本中修正；未知仍未知，不忽略失败用例。",
        "old_scope_status": "保留原V2冻结目标、失败代码、错误日志和范围否定；不回写或把旧版称作完成。",
        "new_downloads": 0, "new_pdf_parses": 0, "new_accounts": 0, "returns_read": False,
        "T11": "NOT_RUN", "goal_achieved": False})
    base_files = [{"path": p.relative_to(BASE).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in sorted(BASE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    save(OUT / "base_input_identity.json", {"at": now(), "base": BASE.relative_to(ROOT).as_posix(), "files": base_files})
    frozen = [p for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "replay_freeze.json", {"at": now(), "base_batch_freeze_sha256": digest(BASE / "batch_freeze.json"),
        "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in frozen]})
    print("已冻结2530字段的并集修复；没有新增下载或PDF重解析。", flush=True)


def assemble_fields():
    original = pd.read_parquet(BASE / "inputs/verified_facts_before.parquet")
    frame = original.copy()
    for column in ["verified_value", "metric_value_cny", "source_raw_value", "source_unit", "source_unit_multiplier", "source_page", "source_locator", "parser_version"]:
        frame["legacy_" + column] = original[column]
    frame["repair_scope"] = "UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"
    frame["repair_admission"] = "LEGACY_STATUS_PRESERVED"
    frame["repair_pdf_path"] = None
    keys = {(str(row.announcement_id), row.metric_id): i for i, row in frame.iterrows()}
    changes = []
    extra_keys = {(r["announcement_id"], r["metric_id"]) for r in read(OUT / "scope_extensions.json")}
    for target in read(OUT / "effective_targets.json"):
        fetch = read(BASE / "fetch_receipts" / (target["sha256"] + ".json"))
        parsed_path = BASE / "parsed_documents" / (target["sha256"] + ".json")
        parsed = read(parsed_path) if parsed_path.exists() else None
        metrics = {m["metric_id"]: m for m in parsed.get("metrics", [])} if parsed else {}
        document_valid = fetch["status"] in {"REUSED_SAME_HASH_PDF", "DOWNLOADED_SAME_HASH_PDF"}
        if document_valid and parsed is not None and parsed.get("status") == "PARSED":
            assert parsed["official_pdf_sha256"] == target["sha256"] == fetch["sha256"]
        for metric in target["target_metrics"]:
            index = keys[(target["announcement_id"], metric)]
            old = original.loc[index]
            current = metrics.get(metric)
            available = document_valid and current is not None
            row = {"announcement_id": target["announcement_id"], "ts_code": target["ts_code"], "report_period": target["report_period"],
                "metric_id": metric, "pdf_sha256": target["sha256"], "old_value_cny": old.verified_value,
                "new_value_cny": np.nan, "fetch_status": fetch["status"], "parsed_source_page": None, "parsed_source_unit": None}
            scope = "CONFIRMED_NON_SUMMARY_ERROR_EXTENSION_1_FIELD" if (target["announcement_id"], metric) in extra_keys else "FROZEN_SUMMARY_PATH_2529_FIELDS"
            frame.at[index, "repair_scope"] = scope
            row["repair_scope"] = scope
            frame.at[index, "repair_pdf_path"] = target["pdf_relative_path"] if document_valid else None
            if available:
                assert current["verification_status"] == "PASS_CURRENT_CELL_SCOPE_UNIT_AND_DUPLICATE_CONSISTENCY"
                expected_scope = "PERIOD_END" if metric == "TOTAL_ASSETS_END" else "YEAR_TO_DATE"
                assert current["statement_scope"] == "CONSOLIDATED_ONLY" and current["value_period_scope"] == expected_scope
                for key, value in current.items():
                    frame.at[index, key] = value
                value = float(current["metric_value_cny"])
                frame.at[index, "verified_value"] = value
                frame.at[index, "amount_recomputed"] = value
                frame.at[index, "parser_version"] = parsed["parser_version"]
                # 原测量的依赖接口仅接受该标志；修正版证据身份另列，不改旧档案标志。
                frame.at[index, "fact_status"] = "VERIFIED_SAVED_ORIGINAL_FACT"
                frame.at[index, "repair_admission"] = "REPARSED_SAME_ORIGINAL_PDF_CURRENT_CELL"
                same = pd.notna(old.verified_value) and abs(value - float(old.verified_value)) <= .011
                row.update({"new_value_cny": value, "status": "REPARSED_UNCHANGED_VALUE" if same else "REPARSED_CHANGED_VALUE",
                    "parsed_source_page": current["source_page"], "parsed_source_unit": current["source_unit"],
                    "new_source_resolution_cny": json.loads(current["source_locator"])["resolution_cny"]})
            else:
                for field in ["verified_value", "amount_recomputed", "metric_value_cny", "source_unit_multiplier"]:
                    frame.at[index, field] = np.nan
                for field in ["source_raw_value", "source_unit", "source_locator"]:
                    frame.at[index, field] = None
                frame.at[index, "fact_status"] = "NO_VIEW_UNRESOLVED_SUMMARY_SOURCE_REPAIR"
                frame.at[index, "repair_admission"] = "NO_VIEW_UNRESOLVED_SUMMARY_SOURCE_REPAIR"
                frame.at[index, "verification_status"] = "NO_VIEW_UNRESOLVED_SUMMARY_SOURCE_REPAIR"
                row["status"] = "NO_VIEW_UNRESOLVED_SOURCE_FIELD"
                row["parse_status"] = parsed.get("decisions", {}).get(metric, {}).get("status") if parsed else None
            changes.append(row)
    changes = pd.DataFrame(changes)
    assert len(changes) == 2530 and not changes.duplicated(["announcement_id", "metric_id"]).any()
    assert frame.loc[frame.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"), "verified_value"].equals(
        original.loc[frame.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"), "verified_value"])
    cases = []
    for case in read(BASE / "inputs/confirmed_contradictions.json"):
        matched = changes[(changes.announcement_id == case["announcement_id"]) & (changes.metric_id == case["metric_id"])].iloc[0]
        cases.append({"announcement_id": case["announcement_id"], "metric_id": case["metric_id"],
            "expected_cny": case["original_page_value_cny"], "observed_cny": matched.new_value_cny,
            "pass": bool(pd.notna(matched.new_value_cny) and abs(matched.new_value_cny - case["original_page_value_cny"]) <= .011)})
    result = {"at": now(), "status": "COMPLETED_SOURCE_REPAIR_WITH_EXPLICIT_UNKNOWNS", "fields_in_version": len(frame),
        "fields_in_repair_scope": len(changes), "fields_in_frozen_summary_scope": 2529, "additional_confirmed_non_summary_fields": 1, "field_status_counts": changes.status.value_counts().to_dict(),
        "unchanged_legacy_fields_outside_scope": int(frame.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH").sum()),
        "known_regressions": cases, "seven_known_regressions_passed": all(c["pass"] for c in cases),
        "all_fields_independently_reverified": False, "historical_first_publication_established": False,
        "new_accounts": 0, "returns_read": False, "goal_achieved": False, "goal_status": "active"}
    scope_row = changes.loc[(changes.announcement_id == "1209410909") & changes.metric_id.eq("OPERATING_CASH_FLOW_YTD")].iloc[0]
    scope_case = {"announcement_id": "1209410909", "metric_id": "OPERATING_CASH_FLOW_YTD", "expected_cny": 585185023.09,
        "observed_cny": scope_row.new_value_cny, "pass": bool(pd.notna(scope_row.new_value_cny) and abs(scope_row.new_value_cny - 585185023.09) <= .011)}
    result["new_scope_regression"] = scope_case
    result["all_confirmed_regressions_passed"] = result["seven_known_regressions_passed"] and scope_case["pass"]
    return frame, changes, result


def replay():
    assert (BASE / "batch_complete.json").exists() and not (OUT / "replay_started.json").exists()
    assert not (BASE / "parser_semantic_invalidation_addendum.json").exists(), "新增原文反证存在，不可将本版作为有效来源重算"
    frozen = read(OUT / "replay_freeze.json")
    for item in frozen["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    for item in read(OUT / "base_input_identity.json")["files"]:
        assert digest(BASE / item["path"]) == item["sha256"], item["path"]
    save(OUT / "replay_started.json", {"at": now(), "replay_freeze_sha256": digest(OUT / "replay_freeze.json")})
    facts, changes, source_result = assemble_fields()
    assert source_result["all_confirmed_regressions_passed"], "已确认错误仍未在最终字段版本中修正"
    facts.to_parquet(OUT / "repaired_verified_facts.parquet", index=False)
    changes.to_parquet(OUT / "field_change_ledger.parquet", index=False)
    changes.to_csv(OUT / "field_change_ledger.csv", index=False, encoding="utf-8-sig")
    save(OUT / "source_repair_result.json", source_result)
    print(f"独立字段版本已形成：{source_result['field_status_counts']}。", flush=True)
    module = load_measurement_module()
    _, events, members, industry, _ = module.validated_sources(INPUTS)
    measured, dependencies = module.measure_events(facts, events, members, industry)
    print("修正版事件依赖已计算，继续按原规则复算逐日覆盖。", flush=True)
    daily, panel = module.daily_coverage(events, measured, members, industry)
    for name, table in [("repaired_member_report_measurements.parquet", measured), ("repaired_formula_dependencies.parquet", dependencies),
                        ("repaired_daily_member_measurements.parquet", panel), ("repaired_daily_coverage.parquet", daily)]:
        table.to_parquet(OUT / name, index=False)
    target = measured.loc[measured.member_at_available]
    annual = daily.assign(year=daily.date.dt.year).groupby("year").agg(trading_days=("date", "size"),
        joint_median=("joint_known_count", "median"), joint_min=("joint_known_count", "min"), joint_max=("joint_known_count", "max"),
        coverage_median=("joint_coverage_nonfinancial", "median")).reset_index()
    annual.to_csv(OUT / "repaired_yearly_daily_coverage.csv", index=False, encoding="utf-8-sig")
    finite = target.loc[target.joint_known]
    output = {"at": now(), "status": "COMPLETED_UNCHANGED_FORMULA_SOURCE_VERSION_REPLAY",
        "target_member_report_events": len(target), "joint_measurable_events_after_source_repair": int(target.joint_known.sum()),
        "L02_measurable_events": int(target.L02_known.sum()), "L04_measurable_events": int(target.L04_known.sum()),
        "yearly_daily_coverage": annual.to_dict("records"), "diagnostic_quantiles": finite[["L02", "L04_change"]].quantile([0, .01, .5, .99, 1]).to_dict(),
        "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "new_accounts": 0, "returns_read": False,
        "goal_achieved": False, "goal_status": "active", "current_market_view": "NO_VIEW", "external_review": "NOT_PERFORMED"}
    save(OUT / "measurement_replay_result.json", output)
    print(json.dumps({"joint_measurable_events": int(target.joint_known.sum()), "new_accounts": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "replay"])
    args = parser.parse_args()
    prepare_replay() if args.action == "prepare" else replay()
