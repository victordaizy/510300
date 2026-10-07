"""分别关闭两种日更学习退出，保留全部其他模块与训练支持路由。"""
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

import research.selected_mix_daily_two_year_v1 as daily
import research.selected_mix_pooled_daily_v1 as pooled
import research.selected_mix_risk_before_band_v1 as execution
import research.post_selection_continuous_replay_v1 as original
from research.selected_mix_reappraisal_v1 import read, save, digest, now, local_import_closure, make_pipeline, MODEL
from research.learned_cycle_exit_v1 import FEATURES, state_values, ExitController
from research.within_cycle_exit_inputs_v1 import WithinCycleExitController
from research.selected_mix_migration_factorial_v1 import account_checks, contrast_interval
from research.strategy_review_diagnostics_v1 import metrics

OUT = ROOT / "reports/research/510300_selected_mix_exit_usage_factorial_v1"
STUDY = "510300_SELECTED_MIX_EXIT_USAGE_FACTORIAL_V1"
CELLS = {"W1_R1": (True, True), "W0_R1": (False, True), "W1_R0": (True, False), "W0_R0": (False, False)}
PRIMARY = "W0_R0"


class PriceBoundaryOnlyController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": "DISABLED_FIXED_COMPONENT_ABLATION",
                "continuation_prediction": None, "learning_fit_origin": pd.NaT,
                "negative_confirmation_count": 0, "learned_exit_requested": False,
                **dict(zip(FEATURES, values))}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("学习退出使用分解已经固定。")
    pooled.verify_sources(pooled.OUT)
    execution.verify_sources(execution.OUT)
    for folder in ["code", "inputs", "results", "accounts"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {"study_id": STUDY, "at": now(), "primary": PRIMARY,
        "question": "同一两年日更系统内，两种学习退出分别以及共同是否有账户增量。",
        "motivation": "限制明显外推使压力夏普0.954变为1.048，但未能分辨支持限制与少用学习退出。",
        "cells": {"W1_R1": "保留周期内模型及普通岭模型的学习退出，直接复用风险先行父账户。",
                  "W0_R1": "关闭周期内模型学习退出，保留普通岭模型学习退出。",
                  "W1_R0": "保留周期内模型学习退出，关闭普通岭模型学习退出。",
                  "W0_R0": "关闭两种学习退出，原价格/止损/期限退出全部继续执行。"},
        "fixed_boundary": "仅控制器是否请求学习退出发生变化。原入场、再入场、价格边界、每日风险估计、支持路由、全部后续组合层和末端尾部规则保持。",
        "support_router": "即使关闭学习退出，仍复用原每日成熟资格作为原支持路由输入；避免把路由变化混进退出作用。该资格不读取未来标签。",
        "training": "复用父共享模型全部两年每日记录和原系数，无新拟合；每日参考风险协方差仍在两年窗口重算。",
        "comparison": "全部日更四格在完整同日账户上比较；两个主效应与交互，20日区块2000次seed20261004，不以最优格替代预先指定的W0_R0。",
        "duplicate_boundary": "此前四格比较旧月度学习与新日更学习，以及旧/新风险分配。本轮的零表示完全不采用该学习退出，而非改用旧月度模型；也不是同时删除多个其他策略模块的简单核心。",
        "period": [daily.START, daily.END], "capital": 200000, "annual_days": 242,
        "targets": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1}, "tail_contract": mandate["account_tail_risk_contract"],
        "new_full_accounts": 6, "reused_full_accounts": 2, "new_internal_dependency_accounts": 66,
        "new_reference_accounts": 0, "new_coefficient_fits": 0, "new_parameter_grid": 0,
        "new_market_collection": False, "goal_achieved": False, "orders_authorized": False,
        "evidence_limit": "全部为重复使用历史上的固定开发比较，不是独立验证。"}
    save(root / "protocol.json", protocol, True)
    files = local_import_closure({Path(__file__)})
    files.update(ROOT / name for name in read(pooled.OUT / "freeze.json")["sources"])
    files.update([pooled.OUT / "results/daily_models.json", daily.OUT / "results/tail_forecasts.json", execution.OUT / "result.json"])
    for cost in ["BASE", "STRESS"]:
        folder = execution.OUT / "accounts" / cost / execution.PRIMARY
        files.update([folder / "ledger.parquet", folder / "decisions.parquet", folder / "checkpoint.json"])
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("两种学习退出的使用分解已固定，全部其他模块保留。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def graph(root, cell, models):
    destination = root / "internal_graphs" / cell
    destination.mkdir(parents=True)
    pipeline = make_pipeline(destination)
    pipeline.variant = "EXIT_USAGE_" + cell
    pipeline.ridge, pipeline.within = models["ridge"], models["within"]
    use_within, use_ridge = CELLS[cell]
    bindings = dict(original.Pipeline.run.__globals__)
    bindings.update(EntryVintageExitController=WithinCycleExitController if use_within else PriceBoundaryOnlyController,
                    ExitController=ExitController if use_ridge else PriceBoundaryOnlyController,
                    continuous_minimum_variance_budget=daily.daily_min_variance,
                    joint_downside_budgets=daily.daily_joint_downside, support_choice=daily.daily_support)
    FunctionType(original.Pipeline.run.__code__, bindings, "固定关闭学习退出的依赖图")(pipeline)
    save(destination / "bindings.json", {"cell": cell, "within_learning_exit": use_within,
         "ridge_learning_exit": use_ridge, "other_interfaces": "保持父两年每日系统", "internal_accounts": len(pipeline.accounts)}, True)
    for node, should_enable in [("ENTRY_VINTAGE_EXIT", use_within), ("REARM_RIDGE_CONTINUOUS", use_ridge)]:
        for (name, _), (_, decisions, _) in pipeline.accounts.items():
            if name == node and not should_enable:
                assert not decisions.learned_exit_requested.fillna(False).any()
    return pipeline


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    models = read(pooled.OUT / "results/daily_models.json")
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    tails = read(daily.OUT / "results/tail_forecasts.json")
    accounts = {}
    for cell in CELLS:
        pipeline = None if cell == "W1_R1" else graph(root, cell, models)
        for cost in cfg["costs"]:
            folder = root / "accounts" / cost / cell
            folder.mkdir(parents=True)
            if cell == "W1_R1":
                source = execution.OUT / "accounts" / cost / execution.PRIMARY
                for name in ["ledger.parquet", "decisions.parquet", "checkpoint.json"]:
                    shutil.copy2(source / name, folder / name)
                ledger, decisions = pd.read_parquet(folder / "ledger.parquet"), pd.read_parquet(folder / "decisions.parquet")
                save(folder / "reused_from.json", {"source": source.relative_to(ROOT).as_posix(), "no_recomputation": True}, True)
            else:
                simulator = FunctionType(execution.simulate.__code__, {**execution.simulate.__globals__, "PRIMARY": cell}, "固定退出使用末端账户")
                ledger, decisions, checkpoint = simulator(pipeline.data, pipeline.div, cfg, pipeline.get(MODEL, cost), tails, cost)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                save(folder / "checkpoint.json", checkpoint, True)
            account_checks(ledger, decisions)
            accounts[cell, cost] = ledger
            measurement = metrics(ledger)
            print(f"退出使用 {cost}/{cell}：夏普{measurement['sharpe']:.6f}，年化{measurement['annual_return']:.2%}，回撤{measurement['max_drawdown']:.2%}。", flush=True)
    measurements, windows = daily.summarize_accounts(root, None, accounts)
    contrasts, rng = [], np.random.default_rng(20261004)
    for cost in cfg["costs"]:
        for name in CELLS:
            pd.testing.assert_series_equal(accounts[name, cost].date, accounts["W1_R1", cost].date)
        r = {name: accounts[name, cost].net_return.to_numpy(float) for name in CELLS}
        effects = {"WITHIN_ON_NO_RIDGE": r["W1_R0"] - r["W0_R0"],
                   "RIDGE_ON_NO_WITHIN": r["W0_R1"] - r["W0_R0"],
                   "INTERACTION": r["W1_R1"] - r["W1_R0"] - r["W0_R1"] + r["W0_R0"],
                   "BOTH_LEARNING_VS_NEITHER": r["W1_R1"] - r["W0_R0"]}
        for name, values in effects.items():
            contrasts.append({"cost": cost, "contrast": name, **contrast_interval(values, rng)})
    primary = next(row for row in measurements if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    verification = {"at": now(), "accounts_checked": 8, "disabled_controllers_never_request_learning_exit": True,
                    "same_daily_training_support_router": True, "new_coefficient_fits": 0, "account_risk_checks": True}
    save(root / "verification.json", verification, True)
    save(root / "result.json", {"study_id": STUDY, "at": now(), "status": "COMPLETED_FIXED_EXIT_USAGE_FACTORIAL",
        "primary": primary, "all_accounts": measurements, "contrasts": contrasts, "verification": verification,
        "new_full_accounts": 6, "reused_full_accounts": 2, "new_internal_accounts": 66, "new_reference_accounts": 0,
        "new_coefficient_fits": 0, "rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in windows if row["cost"] == "STRESS" and row["policy"] == PRIMARY),
        "goal_achieved": False, "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    print("学习退出使用四格完成，全部价格失效及其他模块保持。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="分别关闭两种学习退出的固定四格比较")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
