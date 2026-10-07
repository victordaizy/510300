"""固定五日剩余持有价值，按已成熟状态学习，保留原价格失效边界。"""
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

import research.selected_mix_daily_two_year_v1 as parent
import research.selected_mix_pooled_daily_v1 as pooled
import research.selected_mix_risk_before_band_v1 as execution
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, LATEST, MODEL
from research.learned_cycle_exit_v1 import FEATURES, continuation_label
from research.adaptive_allocation_v1 import normalize_dividends
from research.strategy_review_diagnostics_v1 import metrics

STUDY = "510300_SELECTED_MIX_BOUNDED_VALUE_DAILY_V1"
PRIMARY = "POOLED_BOUNDED5_RISK_BAND10"
OUT = ROOT / "reports/research/510300_selected_mix_bounded_value_daily_v1"
HORIZON = 5
REFERENCES = {"D60_INTRA": parent.OUT / "training_reference",
              **{signal: pooled.OUT / "training_reference" / signal for signal in pooled.SIGNALS if signal != "D60_INTRA"}}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("固定五日价值实验已经登记。")
    for name in ["code", "inputs", "training_reference", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {
        "study_id": STUDY, "at": parent.now(), "primary": PRIMARY,
        "question": "自然退出标签等待整个周期结束，使两年学习延迟；五日结果成熟后即可学习的继续价值能否改善原图。",
        "mechanism": "退出判断预测继续承担近期价格风险相对下个开盘卖出的价值。五日与现行五日尾部预算相同，不扫描期限。",
        "label": "从原状态后的第一个开盘卖出所得，与五个交易日后卖出所得之差，加仅因继续持有新增的股息权益。如果原价格规则实际更早结束，采用该更早退出；已申请价格退出的状态不训练。",
        "label_cost": "沿用自然标签BASE每份卖出所得差与整数参考数量；不重复扣入场沉没成本，末端账户另按两档实际费用结算。",
        "maturity": "每行标签终点严格早于当前决策日即可训练，不要求其原持仓周期已经全部结束。未完成五日且没有更早原退出的状态保留为待成熟。",
        "window": "仍为最近两个日历年，原入场日期及状态日期均不得早于窗口起点；不用窗口外样本。",
        "unchanged": ["原三类价格参考及八状态特征、类别截距和共同斜率", "两年逐日更新、总10周期且各类3周期和100行探索门槛",
                      "原价格退出和学习退出连续两收盘确认", "原内部信号图及每日风险分配",
                      "风险可行目标先行及固定10个百分点调仓带", "原末端尾部合同、交易费用与账户约束"],
        "count_interpretation": "周期数指至少一行标签成熟的已发生参考周期，可仍持仓；不是已完成周期数量或独立路径数。类内各周期总权重相等，三个类别总权重相等。",
        "duplicate_boundary": ["旧第52轮动态继续价值仍等待整个周期结束，并拟合含后续退出的Bellman价值。这里无价值迭代，只学习五日实际可观察结果。",
                               "旧剩余持有迁移用于另一组三形态及期限缩短的退出，本次是原D60依赖图、三类共享斜率与两年每日训练的有限迁移。"],
        "primary_comparator": execution.PRIMARY, "period": [parent.START, parent.END], "capital": 200000,
        "annual_days": 242, "targets": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "new_accounts": 2, "new_internal_dependency_accounts": 22, "new_reference_accounts": 0,
        "comparison": "完整同日账户、20日区块2000次、种子20260928；保留所有年份和滚动两年。",
        "new_parameter_grid": 0, "evidence_class": "已观察历史后的固定开发实验", "orders_authorized": False,
        "new_market_collection": False, "goal_achieved": False}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update(ROOT / name for name in read(pooled.OUT / "freeze.json")["sources"])
    sources.update([parent.OUT / "results/tail_forecasts.json", execution.OUT / "result.json",
                    execution.OUT / "protocol.json", ROOT / "config/510300_learned_cycle_exit_v1.json"])
    for folder in REFERENCES.values():
        sources.update([folder / "decisions.parquet", folder / "cycles.parquet"])
    for cost in ["BASE", "STRESS"]:
        sources.add(execution.OUT / "accounts" / cost / execution.PRIMARY / "ledger.parquet")
    save(root / "freeze.json", {"at": parent.now(), "code_sha256": digest(Path(__file__)),
                                "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("五日继续价值实验已固定：五日已成熟状态可进入两年训练，原价格退出保持。", flush=True)


def verify_sources(root):
    frozen = read(root / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(root / "protocol.json") == frozen["protocol_sha256"]
    for name, expected in frozen["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def build_samples(data, dividends, cutoff=None):
    cfg = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    boundary = len(data) - 1 if cutoff is None else int(cutoff)
    result, pending = [], []
    for code, (signal, folder) in enumerate(REFERENCES.items()):
        decisions = pd.read_parquet(folder / "decisions.parquet")
        cycles = pd.read_parquet(folder / "cycles.parquet")
        for cycle in cycles.to_dict("records"):
            cid = int(cycle["cycle_id"])
            natural = int(data.date.searchsorted(cycle["exit_date"])) if pd.notna(cycle["exit_date"]) else len(data) + 100
            # 在截断回放中，未来才发生的自然退出视为未知；不得藉此纳入未成熟行。
            known_natural = natural if natural <= boundary else len(data) + 100
            held = decisions.loc[decisions.learning_cycle_id.eq(cid) & decisions.requested_quantity.eq(0)
                                 & decisions.origin_index.le(boundary)]
            for row in held.to_dict("records"):
                t = int(row["origin_index"])
                early, late = t + 1, min(t + 1 + HORIZON, known_natural)
                if not np.isfinite([row[k] for k in FEATURES]).all() or late <= early:
                    continue
                item = {"signal": signal, "cycle_id": code * 100000 + cid, "source_cycle_id": cid,
                        "entry_index": int(cycle["entry_index"]), "entry_date": pd.Timestamp(cycle["entry_date"]),
                        "origin_index": t, "origin": data.date.iloc[t], "early_exit_index": early,
                        "reference_quantity": int(cycle["entry_quantity"]), **{k: row[k] for k in FEATURES}}
                if late > boundary:
                    pending.append({**item, "status": "PENDING_FIXED_HORIZON_OR_NATURAL_EXIT"})
                    continue
                target, extra = continuation_label(data, dividends, item["reference_quantity"], early, late,
                                                   cfg["costs"]["BASE"], cfg["tick"])
                result.append({**item, "exit_index": late, "mature_date": data.date.iloc[late],
                               "target": target, "extra_dividend_cny": extra,
                               "holding_intervals": late - early,
                               "ended_before_five": late < t + 1 + HORIZON,
                               "natural_exit_index_if_known": known_natural if known_natural <= boundary else None})
    samples = pd.DataFrame(result).sort_values(["cycle_id", "origin_index"]).reset_index(drop=True)
    samples["sample_id"] = np.arange(len(samples), dtype=int)
    return samples, pd.DataFrame(pending)


def maturity_checks(root, data, dividends, samples):
    comparison_columns = ["signal", "cycle_id", "entry_index", "origin_index", "exit_index", "target", *FEATURES]
    checks = []
    for date in ["2019-12-31", "2022-06-30", "2024-09-30", "2026-08-14"]:
        cutoff = int(data.date.searchsorted(pd.Timestamp(date), side="right")) - 1
        prefix, _ = build_samples(data, dividends, cutoff)
        mature = samples.loc[samples.exit_index.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix[comparison_columns], mature[comparison_columns], check_exact=True)
        checks.append({"cutoff": data.date.iloc[cutoff], "mature_rows": len(mature), "future_natural_exit_hidden": True})
    assert (samples.exit_index > samples.early_exit_index).all()
    assert samples.holding_intervals.between(1, HORIZON).all()
    save(root / "label_maturity_verification.json", {"status": "PASS_TRUNCATED_KNOWN_HISTORY_SAME_MATURE_LABELS",
                                                     "checks": checks}, True)
    return checks


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": parent.now()}, True)
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(parent.END)].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    samples, pending = build_samples(data, dividends)
    samples.to_parquet(root / "training_reference/pooled_samples.parquet", index=False)
    pending.to_parquet(root / "training_reference/pending_states.parquet", index=False)
    label_checks = maturity_checks(root, data, dividends, samples)
    print(f"固定五日标签{len(samples)}行，期末待成熟{len(pending)}行，截断时钟检查通过。", flush=True)
    models = pooled.train(root, data, samples)
    pipeline = parent.graph_run(root, "BOUNDED5", models)
    tails = read(parent.OUT / "results/tail_forecasts.json")
    account_simulator = FunctionType(execution.simulate.__code__, {**execution.simulate.__globals__, "PRIMARY": PRIMARY},
                                    "固定五日价值末端账户")
    accounts, comparisons = {}, []
    rng = np.random.default_rng(20260928)
    for cost in cfg["costs"]:
        ledger, decisions, state = account_simulator(data, dividends, cfg, pipeline.get(MODEL, cost), tails, cost)
        folder = root / "accounts" / cost / PRIMARY
        folder.mkdir(parents=True)
        ledger.to_parquet(folder / "ledger.parquet", index=False)
        decisions.to_parquet(folder / "decisions.parquet", index=False)
        save(folder / "checkpoint.json", state, True)
        accounts[PRIMARY, cost] = ledger
        reference = pd.read_parquet(execution.OUT / "accounts" / cost / execution.PRIMARY / "ledger.parquet")
        comparisons.append({"cost": cost, "left": PRIMARY, "right": execution.PRIMARY,
                            **parent.paired_interval(ledger, reference, rng)})
        m = metrics(ledger)
        print(f"五日继续价值 {cost}：夏普{m['sharpe']:.6f}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, windows = parent.summarize_accounts(root, data, accounts)
    save(root / "results/paired_comparisons.json", comparisons, True)
    checks = pooled.verify(root, data, samples, read(root / "results/daily_models.json"))
    receipts = pd.read_parquet(root / "results/training_receipts.parquet")
    old_cycles = {code * 100000 + int(row.cycle_id): int(data.date.searchsorted(row.exit_date))
                  if pd.notna(row.exit_date) else len(data) + 100
                  for code, folder in enumerate(REFERENCES.values())
                  for row in pd.read_parquet(folder / "cycles.parquet").itertuples()}
    unresolved_rows = 0
    for record in models["within"]:
        if record["status"] == "FIT_COMPLETE":
            subset = samples.loc[samples.sample_id.isin(record["sample_ids"])]
            unresolved_rows += int(subset.cycle_id.map(old_cycles).ge(record["fit_index"]).sum())
    primary = next(row for row in measurements if row["cost"] == "STRESS")
    result = {"study_id": STUDY, "at": parent.now(),
              "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_BOUNDED_HOLDING_VALUE",
              "primary": primary, "all_accounts": measurements, "comparisons": comparisons, "checks": checks,
              "label_maturity_checks": len(label_checks), "mature_label_rows": len(samples), "pending_label_rows": len(pending),
              "reused_training_rows_before_natural_exit": unresolved_rows,
              "eligible_decision_days": int(receipts.eligible_for_fit.sum()),
              "rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in windows if r["cost"] == "STRESS"),
              "new_full_accounts": 2, "new_internal_accounts": 22, "new_reference_accounts": 0,
              "new_independent_observations": 0, "goal_achieved": False,
              "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(root / "result.json", result, True)
    print("固定五日继续价值实验完成，全部结果已保存。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="原共享训练图的固定五日继续价值")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "研究尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
