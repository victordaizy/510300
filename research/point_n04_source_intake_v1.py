"""N04有限来源准入：只核对声明时钟、结构和旧用途，不生成新交易字段。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json

STUDY = "510300_POINT_N04_SOURCE_INTAKE_V1"
OUT = ROOT / "reports/research/510300_point_n04_source_intake_v1"
CURRENT = Path("reports/research/510300_point_current_observation_20261001")
FACTS = Path("reports/research/510300_original_fund_subscription_reports_v1/quarterly_share_flow_facts_v1_1.parquet")
LEGACY_CALENDAR = Path("data/reference/a_share_hs_trading_calendar_2010_2026_v1.parquet")
CURRENT_CALENDAR = Path("data/reference/sse_trade_calendar_2026.csv")
CURRENT_METADATA = Path("data/reference/sse_trade_calendar_2026.metadata.json")
TOP10 = Path("data/raw/constituents/000300_top10_weights_current.parquet")
WEIGHTS = Path("reports/data_quality/510300_csi300_pit_membership_weights_source_remediation_v1.json")
CALENDAR_STUDY = "510300_calendar_liquidity_timing_v1"
QUARTER_STUDY = "510300_original_quarterly_flow_policy_v1"


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sources():
    paths = [Path(__file__).relative_to(ROOT), Path("research/point_account_cashflow_state_v1.py"),
        Path("reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json"),
        Path("reports/research/510300_factor96_program_v1/factor_progress.csv"),
        Path("reports/research/510300_point_next_information_intake_20261002/G1_prior_definition_routing_20261002.json"),
        Path("reports/research/510300_point_next_information_intake_20261002/N04_prior_source_intake_plan.json"),
        FACTS, LEGACY_CALENDAR, CURRENT_CALENDAR, CURRENT_METADATA, TOP10, WEIGHTS,
        Path("data/reference/a_share_hs_trading_calendar_2010_2026_v1_receipt.json"),
        Path("research/original_fund_quarterly_facts_v1_1.py"),
        Path("research/point_forward_calendar_check_v1.py"),
        CURRENT / "results/training_reference/samples.parquet",
        CURRENT / "inputs/candidate_features.parquet",
        CURRENT / "inputs/within_models.json",
        Path("reports/research/510300_point_b01_exit_prediction_v1/protocol.json"),
        Path("reports/research/510300_point_b01_exit_prediction_v1/prediction_summary.json"),
        Path("reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json")]
    base = Path("reports/research/510300_original_fund_subscription_reports_v1")
    paths += [base / "quarterly_facts_v1_1_result.json", base / "quarterly_download_v1_1_result.json"]
    for name in [CALENDAR_STUDY, QUARTER_STUDY]:
        paths += [Path("research") / (name.removeprefix("510300_") + ".py"),
            Path("config") / (name + ".json"), Path("config") / (name + "_manifest.json"),
            Path("reports/research") / name / "result.json"]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    require(not (OUT / "freeze.json").exists(), "N04来源准入已冻结，不覆盖。")
    OUT.mkdir(exist_ok=True)
    card = next(r for r in read_json("reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json")
                if r["id"] == "N04")
    protocol = {"study": STUDY, "frozen_at": now(), "registered_card": card,
        "hypothesis": "原G1季末交互是否已有不同旧用途，以及现有来源能否承载完整原卡。",
        "scope": "只核对列明来源的日期、结构、旧合同和保存裁决；不运行旧策略。",
        "reads": "原样本仅cycle_id/origin/origin_index；日线仅date；季度表仅身份/声明时钟/源状态。",
        "no_target_reads": "本程序不读取原target、价格收益列、份额金额或持仓收益，旧结果仅抽取已保存指标。",
        "calendar_rule": "旧最后5自然日与原卡最后3交易日分开；旧回取日期不是历史事前全年日历证明。",
        "proxy_rule": "原卡机构重仓代理未绑定；基金申赎份额、现今指数前十权重及ETF自身季度收益不能替代。",
        "quantity_rule": "代理和完整末3日时钟未绑定前，N04完整字段及原成员支持为NOT_COMPUTED，不填未知为0或删行。",
        "outputs": "56期声明时钟、1507原点来源声明覆盖、旧用途和有限源裁决；不生成N04交互字段。",
        "old_result_units": "保留旧资金、时期、242日及费用口径，只抽取不重算，不与当前252日账户混排。",
        "candidate_model_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "all_current_history_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "old_failures": "原季度份额/日历失败和所有技术冻结保留；原卡/G1快照NOT_RUN不回写。",
        "review_limit": "列明结构化表及合同的有限复核，不证明所有原PDF没有持仓表或全仓不存在其他代理。"}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"frozen_at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": [{"path": p.as_posix(), "sha256": digest(ROOT / p)} for p in sources()]}, exclusive=True)
    print(json.dumps({"状态": "N04来源准入已冻结，未计算", "来源数": len(sources())}, ensure_ascii=False), flush=True)


def check():
    frozen = read_json(OUT.relative_to(ROOT) / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "N04来源协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "N04冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def derive():
    fact_columns = ["period", "ts_code", "status", "source_usable", "publication_date", "reported_send_date",
                    "feature_available_session", "source_sha256"]
    facts = pd.read_parquet(ROOT / FACTS, columns=fact_columns)
    require(len(facts) == 56 and not facts.period.duplicated().any(), "原56期事实身份不一致。")
    calendar = pd.read_parquet(ROOT / LEGACY_CALENDAR, columns=["trade_date", "is_open", "source", "retrieved_at"])
    dates = pd.DatetimeIndex(calendar.loc[calendar.is_open, "trade_date"])
    require(dates.is_unique and dates.is_monotonic_increasing, "原交易日期重复或倒序。")
    fact_rows = []
    for row in facts.to_dict("records"):
        publication = pd.to_datetime(row["publication_date"])
        sent = pd.to_datetime(row["reported_send_date"])
        available = pd.to_datetime(row["feature_available_session"])
        if pd.notna(publication):
            last = max(publication, sent) if pd.notna(sent) else publication
            later = dates[dates > last]
            recalculated = later[0] if len(later) else pd.NaT
        else:
            recalculated = pd.NaT
        clock_ok = bool(pd.notna(available) and pd.notna(recalculated) and available == recalculated)
        if row["source_usable"]:
            require(clock_ok, "旧季度声明时钟不能复算：" + row["period"])
        fact_rows.append({"period": row["period"], "symbol": row["ts_code"],
            "old_source_status": row["status"], "source_usable_for_old_share_flow": bool(row["source_usable"]),
            "declared_publication_date": publication, "reported_send_date": sent,
            "declared_available_session": available, "recalculated_next_session": recalculated,
            "declared_clock_matches": clock_ok, "source_sha256": row["source_sha256"],
            "admitted_for_institution_proxy": False})
    clocks = pd.DataFrame(fact_rows)
    for column in ["declared_publication_date", "reported_send_date", "declared_available_session", "recalculated_next_session"]:
        clocks[column] = pd.to_datetime(clocks[column]).astype("datetime64[ns]")
    samples = pd.read_parquet(ROOT / CURRENT / "results/training_reference/samples.parquet",
                              columns=["cycle_id", "origin", "origin_index"])
    samples["origin"] = pd.to_datetime(samples.origin).astype("datetime64[ns]")
    market_dates = pd.read_parquet(ROOT / CURRENT / "inputs/candidate_features.parquet", columns=["date"])
    require(len(samples) == 1507, "原自然成员数量不同。")
    pd.testing.assert_index_equal(pd.DatetimeIndex(samples.origin),
        pd.DatetimeIndex(market_dates.date.iloc[samples.origin_index.to_numpy(int)]), check_names=False)
    samples["original_row_number"] = range(len(samples))
    right = clocks.loc[clocks.source_usable_for_old_share_flow, ["period", "declared_available_session"]]
    coverage = pd.merge_asof(samples.sort_values("origin"), right.sort_values("declared_available_session"),
        left_on="origin", right_on="declared_available_session", direction="backward", allow_exact_matches=True)
    coverage = coverage.sort_values("original_row_number").reset_index(drop=True)
    coverage["old_share_flow_source_declared_available"] = coverage.declared_available_session.notna()
    coverage["legacy_calendar_contains_origin"] = coverage.origin.isin(dates)
    current_calendar = pd.read_csv(ROOT / CURRENT_CALENDAR)
    current_dates = pd.DatetimeIndex(pd.to_datetime(current_calendar.trade_date))
    meta = read_json(CURRENT_METADATA)
    require(digest(ROOT / CURRENT_CALENDAR) == meta["sha256"], "现有2026日历文件与metadata身份不同。")
    require(current_dates.is_unique and len(current_dates) == 242 and set(current_dates.year) == {2026},
            "现有2026日历的日期结构不同。")
    coverage["current_2026_calendar_contains_origin"] = coverage.origin.isin(current_dates)
    coverage["institution_proxy_source_status"] = "NO_VIEW_NO_BOUND_INSTITUTION_PROXY_CONTRACT"
    require(not (coverage.declared_available_session > coverage.origin).any(), "来源声明覆盖误读未来公告。")
    top10 = pd.read_parquet(ROOT / TOP10, columns=["date", "index_code", "stock_code", "source", "retrieved_at"])
    weights = read_json(WEIGHTS)["weight_admission"]
    old_contracts, matched_identities = [], []
    for name in [CALENDAR_STUDY, QUARTER_STUDY]:
        config = read_json(Path("config") / (name + ".json"))
        manifest = read_json(Path("config") / (name + "_manifest.json"))
        result = read_json(Path("reports/research") / name / "result.json")
        own_code = Path("research") / (name.removeprefix("510300_") + ".py")
        targets = [own_code] + ([FACTS, LEGACY_CALENDAR] if name == QUARTER_STUDY else [])
        for target in targets:
            entries = [r for r in manifest["files"] if Path(r["path"]) == target]
            require(len(entries) == 1 and entries[0]["sha256"] == digest(ROOT / target),
                    "旧有限来源与其冻结身份不一致：" + target.as_posix())
            matched_identities.append({"manifest": name, "path": target.as_posix(), "matches": True})
        old_contracts.append({"study": name, "terminal": result["status"],
            "capital_cny": config["initial_capital"], "annual_days": config["annual_days"],
            "evaluation_start": config["evaluation_start"], "cutoff": config["data_cutoff"],
            "features": config.get("calendar_columns", config.get("models")),
            "source_clock": config.get("decision_clock", "已公布季报及报送日期之后下一实际交易日"),
            "historical_calendar_first_delivery_proven": config.get("calendar_source_first_delivery_proven"),
            "saved_primary_metrics": [{k: row[k] for k in ["cost", "model", "annualized_return", "net_sharpe", "max_drawdown"]}
                                      for row in result["primary"]]})
    findings = {"quarter_fact_rows": len(clocks), "old_share_flow_usable_rows": int(clocks.source_usable_for_old_share_flow.sum()),
        "old_declared_clock_matches": int(clocks.declared_clock_matches.sum()),
        "old_unusable_periods": clocks.loc[~clocks.source_usable_for_old_share_flow, "period"].tolist(),
        "quarter_fact_schema_role": "原表是份额申赎，不含股票持仓/机构身份；原PDF内容是否另有持仓不由此判断。",
        "legacy_calendar_rows": len(calendar), "legacy_calendar_min": str(dates.min().date()),
        "legacy_calendar_max": str(dates.max().date()), "legacy_calendar_source": calendar.source.drop_duplicates().tolist(),
        "legacy_calendar_retrieved_at": calendar.retrieved_at.drop_duplicates().tolist(),
        "legacy_calendar_historical_first_delivery": "NOT_ESTABLISHED",
        "current_2026_calendar_rows": len(current_dates), "current_2026_calendar_source_status": meta["status"],
        "current_2026_calendar_retrieved_at": meta["retrieved_at"],
        "current_2026_calendar_limit": "2026全年日历不证明2014—2025事前版本，也不定义机构代理。",
        "original_natural_rows": len(samples), "declared_old_report_coverage_rows": int(coverage.old_share_flow_source_declared_available.sum()),
        "legacy_calendar_contains_original_rows": int(coverage.legacy_calendar_contains_origin.sum()),
        "current_2026_calendar_contains_original_rows": int(coverage.current_2026_calendar_contains_origin.sum()),
        "future_declared_report_reads": 0, "top10_rows": len(top10),
        "top10_weight_dates": sorted(pd.to_datetime(top10.date).dt.strftime("%Y-%m-%d").unique().tolist()),
        "top10_receipts": top10.retrieved_at.drop_duplicates().tolist(), "top10_index_codes": top10.index_code.drop_duplicates().tolist(),
        "top10_role": "单期指数前十权重，不是机构重仓股票来源，不向历史回填。",
        "historical_index_weight_admission": {k: weights[k] for k in ["status", "structural_gate_pass", "version_or_as_of_provenance_gate_pass", "snapshot_count_all"]},
        "institution_proxy_contract_bound": False, "admitted_institution_proxy_contracts": 0,
        "full_N04_fields": "NOT_COMPUTED", "full_N04_original_member_support": "NOT_COMPUTED",
        "prediction_models_registered": 0, "old_contracts": old_contracts,
        "selective_old_source_identities": matched_identities}
    return clocks, coverage, findings


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "N04来源准入已开始，不重跑。")
    check()
    write_json(OUT / "RUN_STARTED.json", {"at": now()}, exclusive=True)
    clocks, coverage, findings = derive()
    folder = OUT / "results"
    folder.mkdir(exist_ok=True)
    for name, frame in [("季度事实声明时钟", clocks), ("原自然状态来源声明覆盖", coverage)]:
        frame.to_parquet(folder / (name + ".parquet"), index=False)
        frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {"study": STUDY, "completed_at": now(),
        "status": "NOT_ADMITTED_N04_COMPLETE_INFORMATION_BLOCK_SOURCE_GATE_FAILED",
        "decision": "机构重仓代理未绑定/认证；旧申赎和单期指数权重不能替代，末3交易日的完整事前时钟也未绑定。",
        "source_findings": findings, "frozen_sources_unchanged": check(),
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "new_market_bars": 0, "financial_metrics": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "summary.json", summary, exclusive=True)
    print(json.dumps({"状态": summary["status"], "来源数": summary["frozen_sources_unchanged"],
        "季度声明时钟": findings["old_declared_clock_matches"], "原自然成员": len(coverage),
        "旧申赎声明覆盖": findings["declared_old_report_coverage_rows"],
        "旧日历覆盖": findings["legacy_calendar_contains_original_rows"],
        "完整N04字段": "NOT_COMPUTED", "新拟合和账户": 0}, ensure_ascii=False), flush=True)


def verify():
    require(not (OUT / "saved_output_recomputation_receipt.json").exists(), "N04保存复算已存在，不覆盖。")
    count = check()
    clocks, coverage, findings = derive()
    for name, frame in [("季度事实声明时钟", clocks), ("原自然状态来源声明覆盖", coverage)]:
        pd.testing.assert_frame_equal(frame, pd.read_parquet(OUT / "results" / (name + ".parquet")), check_exact=True)
    summary = read_json(OUT.relative_to(ROOT) / "summary.json")
    require(summary["source_findings"] == findings, "N04保存来源发现与旧合同不同。")
    write_json(OUT / "saved_output_recomputation_receipt.json", {"at": now(),
        "status": "PASS_SAVED_N04_SOURCE_METADATA_AND_PRIOR_CONTRACT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "quarter_clock_rows": len(clocks), "natural_source_coverage_rows": len(coverage),
        "tables_and_findings_exactly_equal": True, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "scope": "声明时钟/来源结构和旧合同复算，不是N04字段、收益或独立策略验证。"}, exclusive=True)
    print("N04声明时钟与来源覆盖保存复算一致；完整交易字段仍未准入。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="N04有限来源及旧用途准入，不生成交易字段")
    parser.add_argument("stage", choices=["freeze", "run", "verify", "check"])
    stage = parser.parse_args().stage
    {"freeze": freeze, "run": run, "verify": verify, "check": check}[stage]()


if __name__ == "__main__":
    main()
