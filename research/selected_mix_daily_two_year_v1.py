"""固定原信号图的两年每日学习迁移：有限对照、尾部预算与完整研究账户。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import FunctionType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.post_selection_continuous_replay_v1 as original_graph
from research.selected_mix_reappraisal_v1 import (LATEST, GRAPH, MODEL, make_pipeline, local_import_closure,
                                                  digest, clean, read, save)
from research.september_monthly_training_v1 import reference_samples
from research.simple_intraday_protection_v1 import make_rules
from research.adaptive_allocation_v1 import normalize_dividends
from research.learned_cycle_exit_v1 import FEATURES, fit_one
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, WithinCycleExitController
from research.joint_downside_reference_pair_inputs_v1 import minimum_joint_downside
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
from research.incremental_selected_intent_mix_inputs_v1 import gate_request as original_outer_request
from research.intraday_overnight_increment_v1 import fill_price
from research.strategy_review_diagnostics_v1 import metrics, cycles

STUDY = "510300_SELECTED_MIX_DAILY_TWO_YEAR_V1"
OUT = ROOT / "reports/research/510300_selected_mix_daily_two_year_v1"
START, END, NEXT = "2020-01-02", "2026-09-16", "2026-09-17"
PRIMARY = "DAILY_TWO_YEAR_MIN5_TAIL"
PROFILE_MINIMUM = {"STRICT10": 10, "EXPLORATORY5": 5}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本轮已冻结，请运行未完成阶段或读取状态。")
    for name in ["code", "inputs", "results", "accounts", "training_reference"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    assert mandate["training_window_calendar_years"] == 2
    assert mandate["model_update_frequency"] == "EVERY_TRADING_DAY"
    save(root / "inputs/previous_mandate.json", mandate, True)
    shutil.copy2(__file__, root / "code/selected_mix_daily_two_year_v1.py")
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    protocol = {
        "study_id": STUDY, "at": now(), "latest_user_instruction": "请继续，直到一切完成",
        "primary": PRIMARY, "assets": ["510300.SH", "CASH_CNY"], "capital": 200000,
        "period": [START, END], "annual_days": 242, "costs": cfg["costs"],
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "question": "原已选信号图迁移到两年每日学习和当前尾部预算后，是否仍具有完整账户优势。",
        "pre_outcome_feasibility": "仅核对已有参考样本日期与数量：两年窗口最多8个成熟周期，旧10周期门槛不能训练。该检查未比较账户收益或标签值。",
        "policies": {
            "ORIGINAL_TAIL": "原保存模型与信号目标，仅接同一尾部预算；属于旧规则研究对照，不符合新训练合同。",
            "DAILY_TWO_YEAR_MIN10_TAIL": "最近两个日历年、逐日更新，保留10周期100行门槛；不合格时按原价格退出与固定预算回退。",
            PRIMARY: "相同两年每日规则，预先固定最低5个完整周期、100行状态；是探索门槛而非可靠优势证明。"},
        "unchanged": ["原60日结构评分及技术价格规则", "两来源85/15权重与外层10个百分点带宽",
                       "原岭惩罚1、八项特征及标准化裁剪5", "原内部1.15倍率、加仓条件与退出确认",
                       "次日开盘、100份、T+1、现金、分红、两档成本"],
        "training": {
            "window": "[决策日减两个日历年, 决策日)，只用入场和全部状态均在窗口内、且退出日严格早于决策日的自然结束周期。",
            "reference": "原D60_INTRA价格退出账户；完整重建标签但逐日先筛成熟周期，无终点强平标签。",
            "minimum_rows": 100, "minimum_cycles": PROFILE_MINIMUM, "maximum_recent_cycles": None,
            "equal_cycle_weight": True, "daily_model_application": "两种学习退出均使用当日合格记录；取消入场日锁住旧系数。",
            "missing": "当天不合格就停止学习退出，保留原价格退出；不携带已经过期的旧模型。",
            "state_memory": "真实账户现金、持仓和风险峰值连续保留；训练观察窗口不重置账户。价格特征保留各自短窗口预热。"},
        "reference_risk_budgets": "原月度协方差与联合下行预算改为每天、最近两日历年，至少242个完整收盘观察；失败当日回到预定等权，不续用过期估计。",
        "tail": {
            "position_target_cap": .5, "five_day_ES95_budget_fraction": .025, "gap_stress_return": -.1,
            "gap_budget_fraction": .05, "drawdown_headroom_share": .5, "stop_drawdown": .1,
            "distribution": "两年内已成熟的次开盘至第六开盘含分红五日收益；匹配当时20日动量正负状态，至少60条，否则使用同窗全部成熟样本；不足60不新增风险。",
            "fees": "新增调仓及未来退出的佣金和不利价格单位滑点计入事前预算。",
            "band": "风险上限优先于10个百分点调仓带；资料未知不能新增买入，仍可按已知风险减仓。",
            "gap_execution": "预算按决策收盘检查；次开盘成交仍受现金、T+1和涨跌停约束。收盘仓位可因价格变化偏离目标，风险预算不保证最大回撤。",
            "stop": "收盘回撤达到10%后下一可卖开盘申请清仓，本轮不重新开启。"},
        "tail_monitor": "每隔5个交易日固定相位记录预测，成熟后才更新。60个非重叠五日结果中超过15%跌破事前5%分位数则报警；本轮只监控，不据结果另调策略。",
        "comparison": "主候选对严格10周期及原模型尾部对照，比较完整同日账户；固定20日区块、2000次、种子20260925计算算术年化收益差区间。",
        "counts": {"signal_graph_reproductions": 2, "internal_dependency_accounts": 44,
                   "new_external_accounts": 6, "reference_account_reconstruction": 1,
                   "new_original_baseline_accounts": 0, "parameter_grid": 0},
        "limits": ["所有历史已被反复研究，迁移结果仍是开发证据。",
                   "5周期门槛是原先已授权的样本放宽探索，不能把状态行数当独立周期。",
                   "不增加宏观或期权资料，不重新优化60日、权重、成本或门槛。",
                   "内部参考账户可有高于50%的仓位，只有末端6个完整账户受当前尾部约束；内部绩效不能晋升为本轮主候选。"],
        "new_data_collection": False, "orders_authorized": False, "goal_achieved": False}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update({LATEST / "candidate_features.parquet", GRAPH,
                    ROOT / "data/reference/510300_dividends.csv",
                    ROOT / "reports/research/510300_incremental_saved_mix_through208/result.json",
                    ROOT / "reports/research/510300_september_monthly_continuation_v1/within_models.json",
                    ROOT / "reports/research/510300_september_monthly_continuation_v1/ridge_models.json",
                    ROOT / "config/510300_incremental_selected_intent_mix_v1.json",
                    ROOT / "config/510300_learned_cycle_exit_v1.json",
                    ROOT / "config/510300_within_cycle_exit_v1.json"})
    sources.update(ROOT / node["configuration"] for node in read(GRAPH)["nodes"])
    for cost in cfg["costs"]:
        sources.update([LATEST / "accounts_run/accounts" / cost / MODEL / "ledger.parquet",
                        LATEST / "accounts_run/accounts" / cost / MODEL / "decisions.parquet"])
    save(root / "freeze.json", {"at": now(), "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)},
                                 "protocol_sha256": digest(root / "protocol.json"), "code_sha256": digest(Path(__file__))}, True)
    mandate.update(latest_user_instruction="请继续，直到一切完成", continuation_requested_at=now(), current_round=STUDY,
                   latest_integrated_experiment=STUDY, current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("两年每日迁移协议已固定：主候选5周期100行，保留10周期及原模型对照，不追加参数搜索。", flush=True)


def verify_sources(root):
    frozen = read(root / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(root / "protocol.json") == frozen["protocol_sha256"]
    for name, expected in frozen["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("冻结来源发生变化：" + name)


def prepare_samples(root, data, dividends):
    cfg = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    frame, ledger, decisions, completed, checkpoint, samples = reference_samples(
        data, dividends, cfg, make_rules(data)["D60_INTRA"], END, NEXT)
    entries = completed[["cycle_id", "entry_index"]].copy()
    samples = samples.merge(entries, on="cycle_id", how="left", validate="many_to_one")
    samples["entry_date"] = pd.to_datetime(data.date.iloc[samples.entry_index.to_numpy(int)].to_numpy())
    samples["sample_id"] = np.arange(len(samples), dtype=int)
    folder = root / "training_reference"
    ledger.to_parquet(folder / "ledger.parquet", index=False)
    decisions.to_parquet(folder / "decisions.parquet", index=False)
    completed.to_parquet(folder / "cycles.parquet", index=False)
    samples.to_parquet(folder / "samples.parquet", index=False)
    save(folder / "checkpoint.json", checkpoint, True)
    print(f"原价格参考账户已重建，保存{len(samples)}条自然结束周期状态；尚未评价新候选账户。", flush=True)
    return samples


def training_at(data, samples, t):
    left = pd.Timestamp(data.date.iloc[t]) - pd.DateOffset(years=2)
    rows = samples.loc[samples.entry_date.ge(left) & samples.origin.ge(left) & samples.exit_index.lt(t)].copy()
    rows = rows.sort_values(["cycle_id", "origin_index"])
    rows["sample_weight"] = 1 / rows.groupby("cycle_id").origin_index.transform("count")
    ids = sorted(int(v) for v in rows.cycle_id.unique())
    return rows, ids, left


def fit_day(data, rows, ids, left, t, minimum_cycles, cfg_ridge, cfg_within):
    eligible = len(ids) >= minimum_cycles and len(rows) >= 100
    missing = int((~np.isfinite(rows[FEATURES].to_numpy(float)).all(axis=1)).sum())
    assert not missing, "训练特征缺失，不能删除样本补救"
    common = {"fit_index": int(t), "fit_origin": str(data.date.iloc[t].date()),
              "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
              "window_start": str(left.date()), "training_cycles": ids, "training_cycle_count": len(ids),
              "training_rows": len(rows), "eligible_for_fit": eligible, "minimum_cycles": minimum_cycles,
              "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
              "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
              "earliest_entry_date": str(rows.entry_date.min().date()) if len(rows) else None,
              "sample_ids": rows.sample_id.to_list(), "sample_membership_hash": frame_hash(rows[["sample_id", "sample_weight"]]),
              "status": "FIT_COMPLETE" if eligible else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS",
              "missing_feature_rows": missing, "failure": None, "model": None}
    ridge, within = deepcopy(common), deepcopy(common)
    if eligible:
        ridge["model"] = fit_one(rows, "RIDGE", cfg_ridge)
        within["model"] = fit_within_cycle_exit(rows, cfg_within)
    return ridge, within


def train(root, data, samples):
    cfg_ridge = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    cfg_within = read(ROOT / "config/510300_within_cycle_exit_v1.json")
    first = int(np.flatnonzero(data.date.ge(cfg_ridge["reference_start"]))[0]) - 1
    models = {key: {"ridge": [], "within": []} for key in PROFILE_MINIMUM}
    receipts = []
    for t in range(first, len(data)):
        rows, ids, left = training_at(data, samples, t)
        for profile, minimum in PROFILE_MINIMUM.items():
            ridge, within = fit_day(data, rows, ids, left, t, minimum, cfg_ridge, cfg_within)
            models[profile]["ridge"].append(ridge)
            models[profile]["within"].append(within)
            receipts.append({"profile": profile, **{k: v for k, v in ridge.items() if k not in {"model", "sample_ids", "training_cycles"}}})
        if (t - first + 1) % 400 == 0:
            print(f"逐日两年训练资格更新：{t - first + 1}/{len(data) - first}个交易日。", flush=True)
    save(root / "results/daily_models.json", models, True)
    pd.DataFrame(receipts).to_parquet(root / "results/training_receipts.parquet", index=False)
    return models


def daily_support(data, models, first):
    result = np.zeros(len(data), bool)
    by_index = {record["fit_index"]: record for record in models}
    for t in range(first - 1, len(data)):
        record = by_index.get(t)
        result[t] = record is not None and record["status"] == "FIT_COMPLETE"
    return result


def daily_min_variance(dates, reference_returns, expert_states, first, window=242):
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(reference_returns, float), np.asarray(expert_states, float)
    rows = []
    for t, date in enumerate(dates):
        weights = np.array([.5, .5])
        left = max(first, int(dates.searchsorted(date - pd.DateOffset(years=2))))
        sample = returns[left:t + 1]
        fitted = False
        if t >= first and len(sample) >= window and np.isfinite(sample).all():
            denominator = float(np.var(sample[:, 0] - sample[:, 1], ddof=1))
            if denominator > 0 and np.std(sample, axis=0, ddof=1).min() > 0:
                covariance = np.cov(sample.T, ddof=1)
                weight = float(np.clip((covariance[1, 1] - covariance[0, 1]) / denominator, 0, 1))
                weights = np.array([weight, 1 - weight])
                fitted = True
        target = float(states[t] @ weights) if t >= first - 1 and np.isfinite(states[t]).all() else np.nan
        rows.append({"date": date, "panic_budget": weights[0], "learned_budget": weights[1],
                     "risk_window_start": dates[left] if left <= t else pd.NaT, "risk_window_observations": max(0, len(sample)),
                     "risk_update_scheduled": t >= first, "risk_estimate_available": fitted, "target": target})
    return pd.DataFrame(rows)


def daily_joint_downside(data, returns, first):
    dates = pd.DatetimeIndex(data.date)
    weights = np.full((len(data), 2), np.nan)
    rows = []
    for t, date in enumerate(dates):
        left = max(first, int(dates.searchsorted(date - pd.DateOffset(years=2))))
        sample = returns[left:t + 1]
        weight, fitted = .5, False
        if t >= first and len(sample) >= 242 and np.isfinite(sample).all():
            weight = float(minimum_joint_downside(sample, .5)["downside_budget"])
            fitted = True
        if t >= first - 1:
            weights[t] = [weight, 1 - weight]
        rows.append({"date": date, "downside_budget": weights[t, 0], "continuous_budget": weights[t, 1],
                     "window_start": dates[left] if left <= t else pd.NaT, "observations": max(0, len(sample)), "updated": fitted})
    return weights, pd.DataFrame(rows)


def graph_run(root, profile, model_records):
    destination = root / "internal_graphs" / profile
    destination.mkdir(parents=True, exist_ok=True)
    pipeline = make_pipeline(destination)
    pipeline.variant = "DAILY_TWO_YEAR_" + profile
    pipeline.ridge, pipeline.within = model_records["ridge"], model_records["within"]
    # 只在本次函数的独立依赖字典中替换四个明确接口，不改旧模块或冻结源码。
    bindings = dict(original_graph.Pipeline.run.__globals__)
    bindings.update(EntryVintageExitController=WithinCycleExitController,
                    continuous_minimum_variance_budget=daily_min_variance,
                    joint_downside_budgets=daily_joint_downside, support_choice=daily_support)
    migrated = FunctionType(original_graph.Pipeline.run.__code__, bindings, "每日两年信号图")
    migrated(pipeline)
    save(destination / "bindings.json", {"original_source": "research/post_selection_continuous_replay_v1.py",
         "original_source_sha256": digest(ROOT / "research/post_selection_continuous_replay_v1.py"),
         "changed_interfaces": ["EntryVintageExitController -> WithinCycleExitController",
                                "continuous_minimum_variance_budget -> daily_min_variance",
                                "joint_downside_budgets -> daily_joint_downside", "support_choice -> daily_support"],
         "models": profile, "accounts": len(pipeline.accounts), "internal_reference_only": True}, True)
    return pipeline


def five_day_labels(data, dividends):
    rows = []
    for t in range(len(data) - 6):
        entry, end = t + 1, t + 6
        entitled = dividends.loc[dividends.record_date.ge(data.date.iloc[entry]) & dividends.record_date.lt(data.date.iloc[end]), "cash_dividend_per_share"].sum()
        value = (data.open.iloc[end] + entitled) / data.open.iloc[entry] - 1
        rows.append({"origin_index": t, "origin": data.date.iloc[t], "exit_index": end, "exit_date": data.date.iloc[end],
                     "momentum_state": int(data.mom20.iloc[t] > 0) if pd.notna(data.mom20.iloc[t]) else -1,
                     "gross_return5": float(value)})
    return pd.DataFrame(rows)


def tail_at(data, labels, t):
    left = pd.Timestamp(data.date.iloc[t]) - pd.DateOffset(years=2)
    eligible = labels.loc[labels.origin.ge(left) & labels.exit_index.lt(t)]
    state = int(data.mom20.iloc[t] > 0) if pd.notna(data.mom20.iloc[t]) else -1
    matched = eligible.loc[eligible.momentum_state.eq(state)] if state >= 0 else eligible.iloc[:0]
    local = matched if len(matched) >= 60 else eligible
    available = len(local) >= 60
    values = local.gross_return5.to_numpy(float)
    tail = np.sort(values)[:max(1, int(np.ceil(len(values) * .05)))]
    return {"origin_index": int(t), "origin": data.date.iloc[t], "window_start": left,
            "available": available, "momentum_state": state, "training_rows": len(local),
            "conditional": len(matched) >= 60, "latest_label_exit_index": int(local.exit_index.max()) if len(local) else -1,
            "label_indices": local.origin_index.to_list(),
            "es95": max(0., -float(tail.mean())) if available else None,
            "q05": float(np.quantile(values, .05)) if available else None}


def risk_plan(shares, nav, peak, price, desired_shares, es95, cost, tick, lot):
    """枚举合法目标份额，交易成本和尾部损失均按决策时价格预算。"""
    quantities = np.arange(0, max(0, int(desired_shares)) + lot, lot, dtype=int)
    buy = fill_price(price, 1, cost, tick)
    sell = fill_price(price, -1, cost, tick)
    changes = quantities - shares
    traded = np.abs(changes)
    trade_price = np.where(changes >= 0, buy, sell)
    commission = np.where(traded > 0, np.maximum(cost["minimum"], traded * trade_price * cost["commission"]), 0.)
    change_cost = commission + traded * np.abs(trade_price - price)
    exit_cost = np.where(quantities > 0, np.maximum(cost["minimum"], quantities * sell * cost["commission"]) + quantities * (price - sell), 0.)
    notional = quantities * price
    tail_budget = nav * .025
    shock_budget = max(0., min(nav * .05, (nav - .9 * peak) * .5))
    tail_loss = notional * es95 + exit_cost + change_cost if es95 is not None else np.full(len(quantities), np.inf)
    gap_loss = notional * .1 + exit_cost + change_cost
    eligible = ((notional <= .5 * (nav - change_cost) + 1e-10) & (tail_loss <= tail_budget + 1e-10)
                & (gap_loss <= shock_budget + 1e-10) & (nav > change_cost))
    choices = np.flatnonzero(eligible)
    picked = int(choices[-1]) if len(choices) else 0
    q = int(quantities[picked]) if len(choices) else 0
    return {"target_shares": q, "risk_plan_feasible": bool(len(choices)), "planned_tail_loss": float(tail_loss[picked]) if es95 is not None else None,
            "planned_gap_loss": float(gap_loss[picked]), "tail_budget": tail_budget, "gap_budget": shock_budget,
            "planned_change_cost": float(change_cost[picked]), "planned_exit_cost": float(exit_cost[picked]),
            "planned_target_exposure": q * price / (nav - change_cost[picked]) if nav > change_cost[picked] else 0.}


def governed_account(data, dividends, cfg, source, tails, name, cost_name):
    cost = cfg["costs"][cost_name]
    peak, stopped = cfg["initial_capital"], False
    tail_map = {r["origin_index"]: r for r in tails}

    def policy(account, price, dummy, config, model, t):
        nonlocal peak, stopped
        nav = account.value(price)
        peak = max(peak, nav)
        drawdown = 1 - nav / peak
        stopped = stopped or drawdown >= .1
        raw = float(source[t])
        if np.isfinite(raw):
            ordinary = original_outer_request(account, price, raw, config, MODEL, True, np.nan)
            desired = max(0, account.shares + int(ordinary["requested_quantity"]))
        else:
            desired = account.shares
        if stopped:
            desired = 0
        estimate = tail_map[t]
        plan = risk_plan(account.shares, nav, peak, price, desired, estimate["es95"], cost, config["tick"], config["lot"])
        q = plan["target_shares"]
        return {"requested_quantity": q - account.shares, "reference_weight": q * price / nav,
                "action": "回撤停止后退出" if stopped else "固定信号经尾部预算决定份额",
                "source_target": raw, "ordinary_target_shares": desired, "shares_before_decision": account.shares,
                "decision_nav": nav, "decision_peak": peak, "decision_drawdown": drawdown,
                "risk_stopped": stopped, "risk_cap_changed_request": q != desired,
                "tail_sample_rows": estimate["training_rows"], "tail_es95": estimate["es95"],
                "tail_q05": estimate["q05"], "conditional_tail": estimate["conditional"], **plan}

    ledger, decisions, state = simulate_indexed_request_account(
        data, dividends, cfg, cost, START, name, targets=np.zeros(len(data)),
        event_mask=np.ones(len(data), bool), request_policy=policy, next_execution_date=NEXT)
    state["risk_governor"] = {"peak": peak, "stopped": stopped}
    return ledger, decisions, state


def paired_interval(left, right, rng):
    a = left.net_return.to_numpy(float) - right.net_return.to_numpy(float)
    n, block, repetitions = len(a), 20, 2000
    means = np.empty(repetitions)
    offsets = np.arange(block)
    for i in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        means[i] = a[indices].mean() * 242
    return {"annual_arithmetic_difference": float(a.mean() * 242),
            "lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975)),
            "block": block, "replications": repetitions, "selection_adjusted": False}


def monitor(tails, labels, first):
    records = pd.DataFrame([{k: v for k, v in r.items() if k != "label_indices"} for r in tails])
    records = records.merge(labels[["origin_index", "exit_index", "exit_date", "gross_return5"]], on="origin_index", how="left")
    records = records.loc[(records.origin_index - (first - 1)) % 5 == 0].copy()
    records = records.loc[records.available & records.exit_index.notna()].sort_values("exit_index").reset_index(drop=True)
    records["breach"] = records.gross_return5 < records.q05
    records["breach_rate60"] = records.breach.rolling(60, min_periods=60).mean()
    records["alert_after_maturity"] = records.breach_rate60 > .15
    return records


def summarize_accounts(root, data, accounts):
    rows, annual, windows, concentration = [], [], [], []
    for (name, cost), ledger in accounts.items():
        m = {"policy": name, "cost": cost, **metrics(ledger)}
        m["historical_point_targets_met"] = m["sharpe"] is not None and m["sharpe"] >= 1.2 and m["annual_return"] >= .1 and m["max_drawdown"] >= -.1
        m["terminal_shares"] = int(ledger.shares.iloc[-1])
        rows.append(m)
        for year, group in ledger.groupby(ledger.date.dt.year):
            first = ledger.index.get_loc(group.index[0])
            capital = 200000. if first == 0 else float(ledger.equity.iloc[first - 1])
            annual.append({"policy": name, "cost": cost, "year": int(year), **metrics(group, capital)})
        for end in range(len(ledger)):
            boundary = ledger.date.iloc[end] - pd.DateOffset(years=2)
            if boundary < ledger.date.iloc[0]:
                continue
            start = int(ledger.date.searchsorted(boundary, side="right"))
            capital = float(ledger.equity.iloc[start - 1]) if start else 200000.
            item = metrics(ledger.iloc[start:end + 1], capital)
            windows.append({"policy": name, "cost": cost, "start": ledger.date.iloc[start], "end": ledger.date.iloc[end],
                            **item, "joint_point_pass": item["sharpe"] is not None and item["sharpe"] >= 1.2 and item["annual_return"] >= .1 and item["max_drawdown"] >= -.1})
        finished, unfinished = cycles(ledger)
        concentration.append({"policy": name, "cost": cost, "complete_cycles": len(finished),
                              "positive_cycles": int(finished.profit.gt(0).sum()) if len(finished) else 0,
                              "largest_profit_share": float(finished.profit.max() / m["profit"]) if len(finished) and m["profit"] > 0 else None,
                              "unfinished_cycle": unfinished})
    save(root / "results/account_metrics.json", rows, True)
    save(root / "results/yearly_metrics.json", annual, True)
    save(root / "results/profit_concentration.json", concentration, True)
    pd.DataFrame(windows).to_parquet(root / "results/rolling_two_years.parquet", index=False)
    return rows, windows


def verify_run(root, data, samples, models, tails):
    counts = {"daily_training_records": 0, "fitted_models": 0, "checked_saved_coefficients": 0,
              "future_and_expired_label_perturbations": 0, "tail_records": 0, "accounts": 0,
              "decision_risk_checks": 0, "risk_driven_exit_infeasibilities": 0}
    cfg_r = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    cfg_w = read(ROOT / "config/510300_within_cycle_exit_v1.json")
    for profile, streams in models.items():
        fitted = []
        for kind, records in streams.items():
            for record in records:
                counts["daily_training_records"] += 1
                if record["latest_exit_index"] is not None:
                    assert record["latest_exit_index"] < record["fit_index"]
                    assert pd.Timestamp(record["earliest_entry_date"]) >= pd.Timestamp(record["window_start"])
                if record["status"] == "FIT_COMPLETE":
                    counts["fitted_models"] += 1
                    if kind == "ridge":
                        fitted.append(record)
        for record in fitted[::max(1, len(fitted) // 6)]:
            t = record["fit_index"]
            rows, ids, left = training_at(data, samples, t)
            rebuilt = fit_day(data, rows, ids, left, t, PROFILE_MINIMUM[profile], cfg_r, cfg_w)
            for kind, value in zip(["ridge", "within"], rebuilt):
                saved_record = streams[kind][t - streams[kind][0]["fit_index"]]
                assert clean(value) == clean(saved_record)
                counts["checked_saved_coefficients"] += 1
            changed = samples.copy()
            excluded = samples.exit_index.ge(t) | samples.entry_date.lt(left)
            changed.loc[excluded, "target"] = 10000.
            again, again_ids, again_left = training_at(data, changed, t)
            assert ids == again_ids and left == again_left
            pd.testing.assert_frame_equal(rows, again, check_exact=True)
            counts["future_and_expired_label_perturbations"] += 1
    for record in tails:
        assert record["latest_label_exit_index"] < record["origin_index"]
        counts["tail_records"] += 1
    for folder in (root / "accounts").glob("*/*"):
        ledger = pd.read_parquet(folder / "ledger.parquet")
        decisions = pd.read_parquet(folder / "decisions.parquet")
        assert ledger.accounting_error.abs().max() < 1e-6
        assert ledger.mark_clock.eq("CLOSE").all()
        assert (ledger.shares % 100 == 0).all() and ledger.cash.ge(-1e-8).all()
        assert pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.loc[data.date.ge(START), "date"]))
        assert pd.DatetimeIndex(decisions.execution_date.iloc[:-1]).equals(pd.DatetimeIndex(ledger.date))
        feasible = decisions.loc[decisions.risk_plan_feasible]
        assert feasible.planned_target_exposure.le(.5 + 1e-10).all()
        assert (feasible.planned_tail_loss <= feasible.tail_budget + 1e-8).all()
        assert (feasible.planned_gap_loss <= feasible.gap_budget + 1e-8).all()
        invalid = decisions.loc[~decisions.risk_plan_feasible]
        assert invalid.target_shares.eq(0).all()
        assert decisions.loc[decisions.source_target.isna(), "requested_quantity"].le(0).all()
        assert decisions.loc[decisions.risk_stopped, "target_shares"].eq(0).all()
        counts["decision_risk_checks"] += len(decisions)
        counts["risk_driven_exit_infeasibilities"] += len(invalid)
        counts["accounts"] += 1
    assert counts["accounts"] == 6
    save(root / "verification.json", {"status": "PASS_TIME_ORDER_SAVED_MODEL_AND_ACCOUNT_RISK_CHECKS", **counts,
                                      "new_independent_observations": 0}, True)
    return counts


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(END)].reset_index(drop=True)
    assert str(data.date.iloc[-1].date()) == END
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    samples = prepare_samples(root, data, dividends)
    models = train(root, data, samples)
    pipelines = {profile: graph_run(root, profile, records) for profile, records in models.items()}
    first = int(np.flatnonzero(data.date.ge(START))[0])
    labels = five_day_labels(data, dividends)
    labels.to_parquet(root / "results/five_day_labels.parquet", index=False)
    tails = [tail_at(data, labels, t) for t in range(first - 1, len(data))]
    save(root / "results/tail_forecasts.json", tails, True)
    observations = monitor(tails, labels, first)
    observations.to_parquet(root / "results/mature_tail_monitor.parquet", index=False)
    accounts = {}
    for cost in cfg["costs"]:
        saved = pd.read_parquet(LATEST / "accounts_run/accounts" / cost / MODEL / "decisions.parquet")
        original = np.full(len(data), np.nan)
        original[saved.origin_index.to_numpy(int)] = saved.reference_weight.to_numpy(float)
        targets = {"ORIGINAL_TAIL": original,
                   "DAILY_TWO_YEAR_MIN10_TAIL": pipelines["STRICT10"].get(MODEL, cost),
                   PRIMARY: pipelines["EXPLORATORY5"].get(MODEL, cost)}
        for name, target in targets.items():
            ledger, decisions, checkpoint = governed_account(data, dividends, cfg, target, tails, name, cost)
            folder = root / "accounts" / cost / name
            folder.mkdir(parents=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            save(folder / "checkpoint.json", checkpoint, True)
            accounts[name, cost] = ledger
            m = metrics(ledger)
            print(f"末端账户 {cost}/{name}：夏普{m['sharpe']}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, windows = summarize_accounts(root, data, accounts)
    rng = np.random.default_rng(20260925)
    comparisons = [{"cost": cost, "left": PRIMARY, "right": control,
                    **paired_interval(accounts[PRIMARY, cost], accounts[control, cost], rng)}
                   for cost in cfg["costs"] for control in ["ORIGINAL_TAIL", "DAILY_TWO_YEAR_MIN10_TAIL"]]
    save(root / "results/paired_comparisons.json", comparisons, True)
    checks = verify_run(root, data, samples, models, tails)
    primary = next(row for row in measurements if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    primary_windows = [row for row in windows if row["policy"] == PRIMARY and row["cost"] == "STRESS"]
    result = {"study_id": STUDY, "at": now(), "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_TWO_YEAR_DAILY_MIGRATION",
              "primary": primary, "all_accounts": measurements, "paired_comparisons": comparisons,
              "checks": checks, "internal_signal_graph_accounts": 44, "reference_reconstruction_accounts": 1,
              "new_complete_accounts": 6, "new_model_fits": checks["fitted_models"],
              "daily_training_qualification_records": checks["daily_training_records"],
              "rolling_two_year_primary_windows": len(primary_windows),
              "rolling_two_year_primary_joint_passes": sum(r["joint_point_pass"] for r in primary_windows),
              "tail_observation_alerts": int(observations.alert_after_maturity.sum()),
              "original_fixed_strategy_changed": False, "new_independent_observations": 0,
              "current_market_view": "NO_VIEW", "goal_achieved": False, "orders_authorized": False}
    save(root / "result.json", result, True)
    print("固定两年每日迁移实验完成，全部对照与失败路径已保留。", flush=True)


def status(root):
    output = {"study": STUDY, "frozen": (root / "freeze.json").exists(), "started": (root / "RUN_STARTED.json").exists(),
              "complete": (root / "result.json").exists(), "goal_achieved": False, "current_market_view": "NO_VIEW"}
    if output["complete"]:
        result = read(root / "result.json")
        output.update(status=result["status"], primary=result["primary"], checks=result["checks"])
    print(json.dumps(clean(output), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="原信号图两年每日学习迁移及账户尾部预算")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "status": status}[args.command](args.out)


if __name__ == "__main__":
    main()
