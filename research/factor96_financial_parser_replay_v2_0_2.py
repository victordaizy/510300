"""统一补入5个旧数值边界候选，以冻结原公式重算财务测量。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research import factor96_financial_parser_replay_v2_0_1 as previous
from research.factor96_earnings_cashflow_measurement_v1 import digest, now, save


ROOT, BASE = previous.ROOT, previous.BASE
SOURCE = previous.OUT
PROBE = ROOT / "reports/research/510300_factor96_legacy_token_boundary_v1"
OUT = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_2"
INPUTS = BASE / "measurement_inputs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_measurement_module():
    spec = importlib.util.spec_from_file_location("unchanged_financial_measurement", OUT / "code/frozen_measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    assert not OUT.exists()
    assert read(SOURCE / "legacy_source_invalidation_addendum.json")["mixed_source_measurement_admitted_to_strategy"] is False
    probe = read(PROBE / "result.json")
    assert probe["candidate_fields"] == probe["same_hash_pdfs"] == 5 and probe["unknown_fields"] == 0
    assert probe["differing_fields"] == 3
    (OUT / "code").mkdir(parents=True)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    shutil.copy2(SOURCE / "code/frozen_measurement.py", OUT / "code/frozen_measurement.py")
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_FACTOR96_FINANCIAL_SCOPE_EXTENSION_V2_0_2",
        "known_before_freeze": "已经看到V2.0.1的财务诊断极值、5项数值边界候选以及同哈希原文的3项明显错误和2项不足1分的差异；尚未计算任何T11收益。",
        "scope": "原2530字段加数值边界扫描的全部5字段，共2535；不只挑3个明显差异。原2112份文档加5份定向同哈希原文，共2117份目标；可用2090份，原27份缺失不重试。",
        "source": "复用V2.0.1重建逻辑及原字段来源；5个新增目标全部使用冻结V2解析器在同哈希原PDF上的已保存当前列结果，不用人工常量替换数据。",
        "unchanged": "原解析器、财务公式、披露日期代理、行业与成员、历史依赖及200日报告年龄全部不变；原V1/V2/V2.0.1错误与结果保留。",
        "unknown_rule": "新增字段若在该原文解析中缺失或身份不一致，旧数值清除并保持未知，不能用旧值回填。",
        "limits": "33138项范围外旧字段仍未全部重新核验；数字边界扫描只覆盖有精确跨度的特定模式。机械重算与已知错误修复不能证明历史首次版本、策略收益或独立验证。",
        "new_http_requests_in_replay": 0, "new_pdf_parses_in_replay": 0,
        "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "new_accounts": 0, "returns_read": False, "goal_achieved": False})
    inputs = [p for folder in [SOURCE, PROBE] for p in sorted(folder.rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts]
    save(OUT / "input_identity.json", {"at": now(), "files": [
        {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in inputs]})
    files = [p for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "replay_freeze.json", {"at": now(), "files": [
        {"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]})
    print("2535字段的新版本已冻结，原失败及原财务公式保持。", flush=True)


def assemble_fields():
    facts, old_changes, result = previous.assemble_fields()
    additions, regression = [], []
    keys = {(str(r.announcement_id), r.metric_id): i for i, r in facts.iterrows()}
    for row in read(PROBE / "candidate_fields.json"):
        aid, metric = row["announcement_id"], row["metric_id"]
        index = keys[(aid, metric)]
        assert facts.at[index, "repair_scope"] == "UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH"
        assert facts.at[index, "verified_value"] == row["old_value_cny"]
        receipt = read(PROBE / "fetch_receipts" / (aid + ".json"))
        assert receipt["status"] == "PASS_SAME_HASH_PDF" and receipt["sha256"] == row["pdf_sha256"]
        assert digest(PROBE / "pdf" / (aid + ".pdf")) == row["pdf_sha256"]
        parsed = read(PROBE / "parsed" / (aid + ".json"))
        current = next((m for m in parsed["metrics"] if m["metric_id"] == metric), None)
        change = {"announcement_id": aid, "ts_code": row["ts_code"], "report_period": row["report_period"],
            "metric_id": metric, "pdf_sha256": row["pdf_sha256"], "old_value_cny": row["old_value_cny"],
            "new_value_cny": np.nan, "fetch_status": "PROBE_SAME_HASH_PDF",
            "parsed_source_page": None, "parsed_source_unit": None,
            "repair_scope": "LEGACY_NUMERIC_BOUNDARY_ALL_5_CANDIDATES"}
        facts.at[index, "repair_scope"] = change["repair_scope"]
        facts.at[index, "repair_pdf_path"] = (PROBE / "pdf" / (aid + ".pdf")).relative_to(ROOT).as_posix()
        if current is not None:
            assert current["verification_status"] == "PASS_CURRENT_CELL_SCOPE_UNIT_AND_DUPLICATE_CONSISTENCY"
            assert current["statement_scope"] == "CONSOLIDATED_ONLY"
            assert current["value_period_scope"] == ("PERIOD_END" if metric == "TOTAL_ASSETS_END" else "YEAR_TO_DATE")
            for name, value in current.items():
                facts.at[index, name] = value
            value = float(current["metric_value_cny"])
            facts.at[index, "verified_value"] = value
            facts.at[index, "amount_recomputed"] = value
            facts.at[index, "parser_version"] = parsed["parser_version"]
            facts.at[index, "fact_status"] = "VERIFIED_SAVED_ORIGINAL_FACT"
            facts.at[index, "repair_admission"] = "REPARSED_SAME_ORIGINAL_PDF_CURRENT_CELL"
            change.update(new_value_cny=value, parsed_source_page=current["source_page"], parsed_source_unit=current["source_unit"],
                status="REPARSED_UNCHANGED_VALUE" if abs(value - row["old_value_cny"]) <= .011 else "REPARSED_CHANGED_VALUE",
                new_source_resolution_cny=json.loads(current["source_locator"])["resolution_cny"])
        else:
            for name in ["verified_value", "amount_recomputed", "metric_value_cny", "source_unit_multiplier"]:
                facts.at[index, name] = np.nan
            for name in ["source_raw_value", "source_unit", "source_locator"]:
                facts.at[index, name] = None
            for name in ["fact_status", "repair_admission", "verification_status"]:
                facts.at[index, name] = "NO_VIEW_UNRESOLVED_SUMMARY_SOURCE_REPAIR"
            change["status"] = "NO_VIEW_UNRESOLVED_SOURCE_FIELD"
        expected = read(PROBE / ("case_" + aid + ".json"))["new_value_cny"]
        regression.append({"announcement_id": aid, "metric_id": metric, "expected_cny": expected,
            "observed_cny": change["new_value_cny"], "pass": bool(current is not None and abs(change["new_value_cny"] - expected) <= .011)})
        additions.append(change)
    changes = pd.concat([old_changes, pd.DataFrame(additions)], ignore_index=True)
    assert len(changes) == 2535 and not changes.duplicated(["announcement_id", "metric_id"]).any()
    outside = facts.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH")
    assert int(outside.sum()) == 33138
    original = pd.read_parquet(BASE / "inputs/verified_facts_before.parquet")
    assert facts.loc[outside, "verified_value"].equals(original.loc[outside, "verified_value"])
    result.update({"at": now(), "fields_in_repair_scope": 2535, "additional_boundary_fields": 5,
        "field_status_counts": changes.status.value_counts().to_dict(), "unchanged_legacy_fields_outside_scope": 33138,
        "boundary_regressions": regression, "all_boundary_regressions_passed": all(r["pass"] for r in regression)})
    result["all_confirmed_regressions_passed"] &= result["all_boundary_regressions_passed"]
    return facts, changes, result


def replay():
    assert not (OUT / "replay_started.json").exists()
    for row in read(OUT / "replay_freeze.json")["files"]:
        assert digest(OUT / row["path"]) == row["sha256"]
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    for row in read(OUT / "input_identity.json")["files"]:
        assert digest(ROOT / row["path"]) == row["sha256"]
    save(OUT / "replay_started.json", {"at": now(), "replay_freeze_sha256": digest(OUT / "replay_freeze.json")})
    facts, changes, source_result = assemble_fields()
    assert source_result["all_confirmed_regressions_passed"]
    facts.to_parquet(OUT / "repaired_verified_facts.parquet", index=False)
    changes.to_parquet(OUT / "field_change_ledger.parquet", index=False)
    changes.to_csv(OUT / "field_change_ledger.csv", index=False, encoding="utf-8-sig")
    save(OUT / "source_repair_result.json", source_result)
    print(f"2535字段版本已形成：{source_result['field_status_counts']}。", flush=True)
    module = load_measurement_module()
    _, events, members, industry, _ = module.validated_sources(INPUTS)
    measured, dependencies = module.measure_events(facts, events, members, industry)
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
    save(OUT / "measurement_replay_result.json", {"at": now(), "status": "COMPLETED_UNCHANGED_FORMULA_SOURCE_VERSION_REPLAY",
        "target_member_report_events": len(target), "joint_measurable_events_after_source_repair": int(target.joint_known.sum()),
        "L02_measurable_events": int(target.L02_known.sum()), "L04_measurable_events": int(target.L04_known.sum()),
        "yearly_daily_coverage": annual.to_dict("records"), "diagnostic_quantiles": finite[["L02", "L04_change"]].quantile([0, .01, .5, .99, 1]).to_dict(),
        "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "new_accounts": 0, "returns_read": False,
        "goal_achieved": False, "goal_status": "active", "current_market_view": "NO_VIEW", "external_review": "NOT_PERFORMED"})
    print("数值边界修复后的原公式复算完成，T11账户仍未运行。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "replay"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else replay()
