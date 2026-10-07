"""V2修复独立报表状态后，按原冻结单季及TTM公式重算覆盖。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.factor96_financial_parser_batch_v2 import OUT, ROOT, read
from research.factor96_earnings_cashflow_measurement_v1 import digest, save, now


INPUTS = OUT / "measurement_inputs"
PRIOR = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"


def load_measurement_module():
    spec = importlib.util.spec_from_file_location("unchanged_earnings_measurement", OUT / "code/frozen_measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_replay():
    assert not (OUT / "replay_freeze.json").exists()
    originals = ["original_facts.parquet", "original_receipt.json", "document_queue.parquet", "report_events.parquet",
                 "membership.parquet", "industry_intervals.parquet", "calendar.parquet"]
    paths = {}
    for name in originals:
        paths[INPUTS / "inputs" / name] = PRIOR / "inputs" / name
    paths[OUT / "code/frozen_measurement.py"] = PRIOR / "code/factor96_earnings_cashflow_measurement_v1.py"
    paths[OUT / "code" / Path(__file__).name] = Path(__file__)
    paths[OUT / "source_evidence/previous_measurement_result.json"] = PRIOR / "result.json"
    for target, source in paths.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    save(OUT / "replay_protocol.json", {"at": now(), "scope": "原固定测量只更换2529条目标来源字段，无经济公式、年龄、同季窗口、单位门槛或入场规则修改。",
        "unknowns": "原文未取得/不同SHA/缺字段/冲突的目标字段在新版本置未知，不能继续读取旧错值；其他字段明确标注沿用旧档案。",
        "clock": "历史名义发布日期与原报告身份保持；本次实际取得时刻另存，仍不声称历史首次版本已证明。",
        "case_gate": "原七条错误以及白云山2020年合并现金流585185023.09元必须通过；任一不能取得均明确修复未完成。",
        "prior_invalid_parser": "V1因独立报表/列头泄漏终止，原重算未运行；V2不得沿用V1取值，全部取得的同哈希目标原文重新解析。",
        "meaning": "只读复算覆盖和字段变化，不形成策略收益；可计算数不等于全数据已通过外部审阅。",
        "new_accounts": 0, "returns_read": False, "goal_achieved": False})
    frozen = list(paths) + [OUT / "replay_protocol.json"]
    save(OUT / "replay_freeze.json", {"at": now(), "batch_freeze_sha256": digest(OUT / "batch_freeze.json"),
        "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in frozen]})
    print("已冻结来源修正后的同公式重算，暂不读取批量处理结果。", flush=True)


def assemble_fields():
    original = pd.read_parquet(OUT / "inputs/verified_facts_before.parquet")
    frame = original.copy()
    for column in ["verified_value", "metric_value_cny", "source_raw_value", "source_unit", "source_unit_multiplier", "source_page", "source_locator", "parser_version"]:
        frame["legacy_" + column] = original[column]
    frame["repair_scope"] = "UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"
    frame["repair_admission"] = "LEGACY_STATUS_PRESERVED"
    frame["repair_pdf_path"] = None
    keys = {(str(row.announcement_id), row.metric_id): i for i, row in frame.iterrows()}
    changes = []
    for target in read(OUT / "batch_targets.json"):
        fetch = read(OUT / "fetch_receipts" / (target["sha256"] + ".json"))
        parsed_path = OUT / "parsed_documents" / (target["sha256"] + ".json")
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
            frame.at[index, "repair_scope"] = "FROZEN_SUMMARY_PATH_2529_FIELDS"
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
    assert len(changes) == 2529 and not changes.duplicated(["announcement_id", "metric_id"]).any()
    assert frame.loc[frame.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"), "verified_value"].equals(
        original.loc[frame.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"), "verified_value"])
    cases = []
    for case in read(OUT / "inputs/confirmed_contradictions.json"):
        matched = changes[(changes.announcement_id == case["announcement_id"]) & (changes.metric_id == case["metric_id"])].iloc[0]
        cases.append({"announcement_id": case["announcement_id"], "metric_id": case["metric_id"],
            "expected_cny": case["original_page_value_cny"], "observed_cny": matched.new_value_cny,
            "pass": bool(pd.notna(matched.new_value_cny) and abs(matched.new_value_cny - case["original_page_value_cny"]) <= .011)})
    result = {"at": now(), "status": "COMPLETED_SOURCE_REPAIR_WITH_EXPLICIT_UNKNOWNS", "fields_in_version": len(frame),
        "fields_in_frozen_summary_scope": len(changes), "field_status_counts": changes.status.value_counts().to_dict(),
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
    assert (OUT / "batch_complete.json").exists() and not (OUT / "replay_started.json").exists()
    assert not (OUT / "parser_semantic_invalidation_addendum.json").exists(), "新增原文反证存在，不可将本版作为有效来源重算"
    frozen = read(OUT / "replay_freeze.json")
    for item in frozen["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    save(OUT / "replay_started.json", {"at": now(), "replay_freeze_sha256": digest(OUT / "replay_freeze.json")})
    facts, changes, source_result = assemble_fields()
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
