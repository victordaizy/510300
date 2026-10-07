"""固定旧诊断区间，检查单项外推限制在不同历史环境中的表现。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.selected_mix_daily_two_year_v1 as daily
import research.selected_mix_pooled_daily_v1 as pooled
import research.selected_mix_risk_before_band_v1 as execution
import research.selected_mix_support_envelope_daily_v1 as support
import research.post_selection_continuous_replay_v1 as original
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, make_pipeline, MODEL, now
from research.learned_cycle_exit_v1 import ExitController
from research.within_cycle_exit_inputs_v1 import WithinCycleExitController
from research.simple_intraday_protection_v1 import make_rules
from research.selected_mix_migration_factorial_v1 import account_checks
from research.strategy_review_diagnostics_v1 import metrics

OUT = ROOT / "reports/research/510300_selected_mix_support_earlier_diagnostic_v1"
STUDY = "510300_SELECTED_MIX_SUPPORT_EARLIER_DIAGNOSTIC_V1"
START, END, NEXT = "2015-01-05", "2019-12-31", "2020-01-02"
POLICIES = ["UNCHANGED_DAILY", "SUPPORT_ENVELOPE"]
PRIMARY = "SUPPORT_ENVELOPE"


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("旧区间反例检查已经固定。")
    support.verify_sources(support.OUT)
    for folder in ["code", "inputs", "results", "accounts"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {"study_id": STUDY, "at": now(), "primary": PRIMARY,
        "question": "单项外推限制相对不限制外推的两年日更系统，是否在既定2015-2019环境也有同向改善。",
        "period": [START, END], "period_source": "原项目早期诊断固定为2015-01-05至2019-12-31，不根据本轮走势挑选边界。",
        "evidence": "该段历史曾用于旧研究，不是独立留出；只检验反例，不拼接到2020年起主账户，不据它选择新的规则。",
        "unchanged_rule": "完全沿用支持范围实验的两年日更记录、八项范围、原系数、价格退出、全部组合层和末端尾部预算。",
        "account_initialization": "两个末端账户各在2015-01-05以20万元现金开始。原2013参考节点保留原始开始日，原2020主账户节点统一改在本诊断起点开始。",
        "clock": "全量保存模型中仅选择当日记录，成熟退出索引严格早于当日；已保存范围仅取当日训练成员。五日风险分布按同一已冻结函数在每个历史原点重建，不能借用2020年后的估计。",
        "terminal": "2019-12-31按正常收盘结算，保留实际未平仓份额，不使用原旧诊断的人工终点开盘平仓。",
        "primary_comparator": "UNCHANGED_DAILY", "capital": 200000, "annual_days": 242,
        "targets": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "tail_contract": mandate["account_tail_risk_contract"],
        "new_full_accounts": 4, "new_internal_accounts": 44, "new_reference_accounts": 0,
        "reference_replays_included_in_internal_count": True, "new_coefficient_fits": 0, "new_parameter_grid": 0,
        "statistics": "同日完整账户收益差，20日区块2000次，种子20261005；两段的增量分开披露。",
        "new_market_collection": False, "orders_authorized": False, "goal_achieved": False}
    save(root / "protocol.json", protocol, True)
    files = local_import_closure({Path(__file__)})
    files.update(ROOT / name for name in read(support.OUT / "freeze.json")["sources"])
    files.update([support.OUT / "results/daily_support_envelopes.json", support.OUT / "result.json"])
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("2015-2019固定反例区间已登记，规则不变，历史不拼接。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def graph(root, policy, models):
    destination = root / "internal_graphs" / policy
    destination.mkdir(parents=True)
    pipeline = make_pipeline(destination)
    pipeline.variant = "EARLIER_" + policy
    pipeline.data = pipeline.data.loc[pipeline.data.date.le(END)].reset_index(drop=True)
    pipeline.start, pipeline.next_date = START, NEXT
    pipeline.first = int(np.flatnonzero(pipeline.data.date.ge(START))[0])
    pipeline.cfg["evaluation_start"], pipeline.cfg["data_cutoff"] = START, END
    for node in pipeline.graph.values():
        if node["replay_start"] == daily.START:
            node["replay_start"] = START
        assert pd.Timestamp(node["replay_start"]) <= pd.Timestamp(START)
    pipeline.ridge, pipeline.within = models["ridge"], models["within"]
    pipeline.session = make_rules(pipeline.data)["D60_INTRA"]
    bindings = dict(original.Pipeline.run.__globals__)
    enabled = policy == "SUPPORT_ENVELOPE"
    bindings.update(EntryVintageExitController=support.SupportEnvelopeExitController if enabled else WithinCycleExitController,
                    ExitController=support.SupportEnvelopeExitController if enabled else ExitController,
                    continuous_minimum_variance_budget=daily.daily_min_variance,
                    joint_downside_budgets=daily.daily_joint_downside, support_choice=daily.daily_support)
    FunctionType(original.Pipeline.run.__code__, bindings, "固定早期区间依赖图")(pipeline)
    for (node, _), (ledger, decisions, _) in pipeline.accounts.items():
        assert ledger.date.iloc[-1] == pd.Timestamp(END)
        assert ledger.date.iloc[0] == pd.Timestamp(pipeline.graph[node]["replay_start"])
        if "learning_fit_origin" in decisions:
            usable = decisions.learning_fit_origin.notna()
            assert pd.to_datetime(decisions.loc[usable, "learning_fit_origin"]).le(decisions.loc[usable, "origin"]).all()
    save(destination / "bindings.json", {"start": START, "end": END, "policy": policy, "accounts": len(pipeline.accounts),
         "reference_nodes_keep_original_start": True, "main_nodes_reset_to_fixed_earlier_start": True}, True)
    return pipeline


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    models = read(pooled.OUT / "results/daily_models.json")
    envelopes = {r["fit_index"]: r for r in read(support.OUT / "results/daily_support_envelopes.json")}
    used_records = 0
    for kind in models:
        for record in models[kind]:
            if record["status"] == "FIT_COMPLETE":
                record["marginal_support"] = envelopes[record["fit_index"]]
                if START <= record["fit_origin"] <= END:
                    assert record["latest_exit_index"] < record["fit_index"]
                    assert pd.Timestamp(record["earliest_entry_date"]) >= pd.Timestamp(record["window_start"])
                    used_records += 1
    accounts, tail_records = {}, None
    proxy = SimpleNamespace(**{**vars(daily), "START": START, "END": END, "NEXT": NEXT})
    for policy in POLICIES:
        pipeline = graph(root, policy, models)
        if tail_records is None:
            labels = daily.five_day_labels(pipeline.data, pipeline.div)
            tail_records = [daily.tail_at(pipeline.data, labels, t) for t in range(pipeline.first - 1, len(pipeline.data))]
            save(root / "results/earlier_tail_forecasts.json", tail_records, True)
            for record in tail_records:
                selected = labels.loc[labels.origin_index.isin(record["label_indices"])]
                assert selected.exit_index.lt(record["origin_index"]).all()
                assert selected.origin.ge(pd.Timestamp(record["origin"]) - pd.DateOffset(years=2)).all()
        simulator = FunctionType(execution.simulate.__code__, {**execution.simulate.__globals__, "PRIMARY": policy, "parent": proxy}, "固定早期区间末端账户")
        for cost in pipeline.cfg["costs"]:
            ledger, decisions, checkpoint = simulator(pipeline.data, pipeline.div, pipeline.cfg, pipeline.get(MODEL, cost), tail_records, cost)
            folder = root / "accounts" / cost / policy
            folder.mkdir(parents=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            save(folder / "checkpoint.json", checkpoint, True)
            account_checks(ledger, decisions)
            assert ledger.date.iloc[0] == pd.Timestamp(START) and ledger.date.iloc[-1] == pd.Timestamp(END)
            accounts[policy, cost] = ledger
            m = metrics(ledger)
            print(f"早期反例 {cost}/{policy}：夏普{m['sharpe']:.6f}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, windows = daily.summarize_accounts(root, None, accounts)
    comparisons, rng = [], np.random.default_rng(20261005)
    for cost in ["BASE", "STRESS"]:
        comparisons.append({"cost": cost, "left": PRIMARY, "right": "UNCHANGED_DAILY",
                            **daily.paired_interval(accounts[PRIMARY, cost], accounts["UNCHANGED_DAILY", cost], rng)})
    primary = next(row for row in measurements if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    main = next(row for row in read(support.OUT / "result.json")["comparisons"] if row["cost"] == "STRESS")
    early = next(row for row in comparisons if row["cost"] == "STRESS")
    verification = {"at": now(), "accounts_checked": 4, "model_records_checked_in_fixed_period": used_records,
                    "risk_forecasts_checked": len(tail_records), "reference_and_main_start_dates_checked": True,
                    "last_day_close_valuation_no_forced_exit": True, "new_coefficient_fits": 0}
    save(root / "verification.json", verification, True)
    save(root / "result.json", {"study_id": STUDY, "at": now(), "status": "COMPLETED_FIXED_EARLIER_COUNTEREVIDENCE",
        "primary": primary, "all_accounts": measurements, "comparisons": comparisons, "verification": verification,
        "stress_increment_point_estimate_same_sign_across_periods": bool(main["annual_arithmetic_difference"] * early["annual_arithmetic_difference"] > 0),
        "main_period_increment": main, "early_period_increment": early, "new_full_accounts": 4,
        "new_internal_accounts": 44, "new_reference_accounts": 0, "new_coefficient_fits": 0,
        "rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in windows if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
        "goal_achieved": False, "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    print("早期反例检查完成，两个区间分别保留，没有拼接或改变规则。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="固定2015-2019诊断区间的支持范围检查")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
