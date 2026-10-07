"""固定两因素比较，区分退出学习迁移和参考风险分配迁移的账户影响。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.post_selection_continuous_replay_v1 as original
import research.selected_mix_daily_two_year_v1 as daily
import research.selected_mix_pooled_daily_v1 as pooled
import research.selected_mix_risk_before_band_v1 as execution
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, make_pipeline, LATEST, MODEL
from research.within_cycle_exit_inputs_v1 import WithinCycleExitController
from research.adaptive_allocation_v1 import normalize_dividends
from research.strategy_review_diagnostics_v1 import metrics

STUDY = "510300_SELECTED_MIX_MIGRATION_FACTORIAL_V1"
OUT = ROOT / "reports/research/510300_selected_mix_migration_factorial_v1"
CELLS = {"E0_R0": (False, False), "E1_R0": (True, False),
         "E0_R1": (False, True), "E1_R1": (True, True)}
LABELS = {"E0_R0": "原退出与原分配", "E1_R0": "日更退出与原分配",
          "E0_R1": "原退出与日更分配", "E1_R1": "日更退出与日更分配"}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("迁移分解已固定，不能改写比较。")
    for name in ["code", "inputs", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {
        "study_id": STUDY, "at": daily.now(), "latest_user_instruction": "不找到不中止",
        "question": "已观察到的两年日更下降，主要来自退出学习块、参考风险分配块，还是两者交互。",
        "cells": {k: {"daily_exit_block": v[0], "daily_risk_block": v[1]} for k, v in CELLS.items()},
        "exit_block": "E0为原月度模型和入场锁定周期内系数；E1为已保存三类共享两年每日模型、当日应用及按当日拟合资格路由。",
        "risk_block": "R0为原月度242日协方差和联合下行参考分配；R1为已实现最近两日历年每日分配。其余价格与风险公式保持。",
        "common_account": "四格均接同一50%目标上限、五日ES预算、跳空及回撤余量，并统一先算风险可行目标再应用10个百分点带宽。",
        "contract_status": "只有E1_R1完整符合当前两年日更合同，且直接复用上一轮账户；其他三格仅用于结构诊断，不按结果晋升或恢复旧原版。",
        "period": [daily.START, daily.END], "capital": 200000, "annual_days": 242,
        "costs": read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")["costs"],
        "primary_contrasts": ["E1_R0-E0_R0", "E0_R1-E0_R0", "E1_R1-E1_R0-E0_R1+E0_R0"],
        "statistics": "完整同日账户算术年化收益差，20日循环区块2000次、种子20260929；区间描述，不声明完整家族选择校正。",
        "attribution": "两个结构块同时包含多个原先已声明的接口，比较不将影响归因于某个单独特征或某个参数。账户现金与风险反馈属于各反事实路径的一部分。",
        "new_model_fits": 0, "new_internal_graph_accounts": 44, "new_terminal_accounts": 6,
        "reused_terminal_accounts": 2, "new_parameter_grid": 0,
        "goal_achieved": False, "new_market_collection": False, "orders_authorized": False}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update(ROOT / name for name in read(pooled.OUT / "freeze.json")["sources"])
    sources.update([pooled.OUT / "results/daily_models.json", daily.OUT / "results/tail_forecasts.json",
                    execution.OUT / "result.json"])
    for cost in ["BASE", "STRESS"]:
        sources.add(daily.OUT / "accounts" / cost / "ORIGINAL_TAIL/decisions.parquet")
        for filename in ["ledger.parquet", "decisions.parquet", "checkpoint.json"]:
            sources.add(execution.OUT / "accounts" / cost / execution.PRIMARY / filename)
    save(root / "freeze.json", {"at": daily.now(), "code_sha256": digest(Path(__file__)),
                                "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    mandate.update(latest_user_instruction="不找到不中止", continuation_requested_at=daily.now(), current_round=STUDY,
                   latest_integrated_experiment=STUDY, current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("退出学习与参考风险分配的四格比较已固定，不按诊断结果挑选旧规则。", flush=True)


def verify_sources(root):
    frozen = read(root / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(root / "protocol.json") == frozen["protocol_sha256"]
    for name, expected in frozen["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def hybrid_graph(root, cell, models):
    exit_new, risk_new = CELLS[cell]
    destination = root / "internal_graphs" / cell
    destination.mkdir(parents=True)
    pipeline = make_pipeline(destination)
    pipeline.variant = "MIGRATION_FACTORIAL_" + cell
    bindings = dict(original.Pipeline.run.__globals__)
    interfaces = []
    if exit_new:
        pipeline.ridge, pipeline.within = models["ridge"], models["within"]
        bindings.update(EntryVintageExitController=WithinCycleExitController, support_choice=daily.daily_support)
        interfaces.extend(["saved_daily_pooled_exit_models", "WithinCycleExitController", "daily_support"])
    if risk_new:
        bindings.update(continuous_minimum_variance_budget=daily.daily_min_variance,
                        joint_downside_budgets=daily.daily_joint_downside)
        interfaces.extend(["daily_min_variance", "daily_joint_downside"])
    FunctionType(original.Pipeline.run.__code__, bindings, "迁移分解依赖图")(pipeline)
    save(destination / "bindings.json", {"cell": cell, "interfaces": interfaces,
                                         "accounts": len(pipeline.accounts), "diagnostic_only": True}, True)
    return pipeline


def account_checks(ledger, decisions):
    nav = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable
    np.testing.assert_allclose(nav, ledger.equity, atol=1e-7, rtol=0)
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.cash.ge(-1e-8).all()
    assert ledger.mark_clock.eq("CLOSE").all()
    np.testing.assert_array_equal(ledger.requested_quantity, decisions.requested_quantity.iloc[:-1])
    feasible = decisions.loc[decisions.risk_plan_feasible]
    assert feasible.planned_target_exposure.le(.5 + 1e-10).all()
    assert (feasible.planned_tail_loss <= feasible.tail_budget + 1e-8).all()
    assert (feasible.planned_gap_loss <= feasible.gap_budget + 1e-8).all()


def contrast_interval(values, rng):
    n, block = len(values), 20
    means = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        means.append(float(values[indices].mean() * 242))
    return {"annual_arithmetic_difference": float(values.mean() * 242),
            "lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975)),
            "selection_adjusted": False}


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": daily.now()}, True)
    models = read(pooled.OUT / "results/daily_models.json")
    hybrids = {cell: hybrid_graph(root, cell, models) for cell in ["E1_R0", "E0_R1"]}
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(daily.END)].reset_index(drop=True)
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    tails = read(daily.OUT / "results/tail_forecasts.json")
    accounts = {}
    for cell in CELLS:
        for cost in cfg["costs"]:
            folder = root / "accounts" / cost / cell
            folder.mkdir(parents=True)
            if cell == "E1_R1":
                source_folder = execution.OUT / "accounts" / cost / execution.PRIMARY
                for filename in ["ledger.parquet", "decisions.parquet", "checkpoint.json"]:
                    shutil.copy2(source_folder / filename, folder / filename)
                ledger, decisions = pd.read_parquet(folder / "ledger.parquet"), pd.read_parquet(folder / "decisions.parquet")
                save(folder / "reused_from.json", {"path": source_folder.relative_to(ROOT).as_posix(),
                                                   "same_daily_models_and_risk_and_band": True}, True)
            else:
                if cell == "E0_R0":
                    old = pd.read_parquet(daily.OUT / "accounts" / cost / "ORIGINAL_TAIL/decisions.parquet")
                    source = np.full(len(data), np.nan)
                    source[old.origin_index.to_numpy(int)] = old.source_target.to_numpy(float)
                else:
                    source = hybrids[cell].get(MODEL, cost)
                simulator = FunctionType(execution.simulate.__code__, {**execution.simulate.__globals__, "PRIMARY": cell}, "分解末端账户")
                ledger, decisions, state = simulator(data, dividends, cfg, source, tails, cost)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                save(folder / "checkpoint.json", state, True)
            account_checks(ledger, decisions)
            accounts[cell, cost] = ledger
            m = metrics(ledger)
            print(f"{LABELS[cell]} {cost}：夏普{m['sharpe']:.6f}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, _ = daily.summarize_accounts(root, data, accounts)
    contrasts = []
    rng = np.random.default_rng(20260929)
    for cost in cfg["costs"]:
        series = {cell: accounts[cell, cost].net_return.to_numpy(float) for cell in CELLS}
        for name, values in {
            "EXIT_BLOCK_ON_ORIGINAL_RISK": series["E1_R0"] - series["E0_R0"],
            "RISK_BLOCK_ON_ORIGINAL_EXIT": series["E0_R1"] - series["E0_R0"],
            "INTERACTION": series["E1_R1"] - series["E1_R0"] - series["E0_R1"] + series["E0_R0"],
            "EXIT_SHAPLEY_DESCRIPTIVE": .5 * (series["E1_R0"] - series["E0_R0"] + series["E1_R1"] - series["E0_R1"]),
            "RISK_SHAPLEY_DESCRIPTIVE": .5 * (series["E0_R1"] - series["E0_R0"] + series["E1_R1"] - series["E1_R0"]),
        }.items():
            contrasts.append({"cost": cost, "contrast": name, **contrast_interval(values, rng)})
    save(root / "results/contrasts.json", contrasts, True)
    result = {"study_id": STUDY, "at": daily.now(), "status": "COMPLETED_FIXED_BLOCK_DIAGNOSTIC",
              "all_accounts": measurements, "contrasts": contrasts, "new_models": 0,
              "new_terminal_accounts": 6, "reused_terminal_accounts": 2, "new_internal_accounts": 44,
              "accounts_checked": 8, "contract_eligible_cell": "E1_R1", "goal_achieved": False,
              "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(root / "result.json", result, True)
    report = ["退出学习和参考风险分配的固定四格比较已完成。所有格使用相同末端尾部预算与调仓带。", ""]
    for row in measurements:
        if row["cost"] == "STRESS":
            report.append(f"- {LABELS[row['policy']]}：压力夏普{row['sharpe']:.3f}，年化{row['annual_return']:.2%}，回撤{abs(row['max_drawdown']):.2%}。")
    report += ["", "结构块的压力账户算术年化影响（相对原退出和原参考分配）：", ""]
    for row in contrasts:
        if row["cost"] == "STRESS":
            report.append(f"- {row['contrast']}：{row['annual_arithmetic_difference']*100:+.3f}个百分点；95%区间[{row['lower_95']*100:+.3f}, {row['upper_95']*100:+.3f}]个百分点。")
    report += ["", "只有日更退出与日更分配这一格符合当前完整合同，它复用前轮账户。其他格用于解释迁移差异，不能按历史表现晋升，也不能归因于某个单一特征。", "",
               "本轮新增6条末端和44条内部依赖账户，未重新训练模型。两因素影响含账户路径反馈；区间未经完整研究家族选择校正。全部历史已被反复使用，未增加独立验证证据。", "",
               "目标未完成，采集仍暂停，没有期权或交易订单。"]
    (root / "研究结论.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("固定结构分解已完成，全部反事实账户保留。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="两年日更迁移的固定两因素诊断")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "研究尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
