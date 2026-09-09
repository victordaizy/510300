"""固定事前状态，对保存账户的收益差作完整日历归因和同时区间。"""
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_saved_state_failure_diagnostic_v1"
CONFIG = ROOT / "config/510300_saved_state_failure_diagnostic_v1.json"
SOURCE = ROOT / "reports/research/510300_heikin_price_state_v1"
STATES = {0: "趋势未向上、波动未上升", 1: "趋势未向上、波动上升", 2: "趋势向上、波动未上升", 3: "趋势向上、波动上升"}
CONTRASTS = {
    "143_VS_BUY_HOLD": ("TREND_NOISE_REFERENCE_BLEND", "BUY_HOLD", "143相对买入持有"),
    "150_REDUCE_VS143": ("EPISODE_BUDGET_NONINCREASING", "TREND_NOISE_REFERENCE_BLEND", "150只减预算相对143"),
    "151_CONFIRM_VS143": ("HEIKIN_CONFIRMED_REFERENCE", "TREND_NOISE_REFERENCE_BLEND", "151平均K线确认相对143"),
}


def freeze():
    require(not CONFIG.exists(), "保存状态诊断已经冻结")
    old_path = ROOT / "config/510300_heikin_price_state_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    cfg = {key: old[key] for key in ["features", "dividends", "evaluation_start", "data_cutoff", "earlier_start", "earlier_terminal", "initial_capital", "annual_days", "costs"]}
    cfg.update(study_id="510300_SAVED_STATE_FAILURE_DIAGNOSTIC_V1", registered_at=now(), trend_window=120, volatility_windows=[20, 60],
        state_clock="PREVIOUS_TRADING_DAY_1505", blocks=[20, 60], repetitions=2000, random_seed=20260909,
        simultaneous_comparisons=48, policy_candidates=0, new_accounts=0, new_model_fits=0, diagnostic_only=True,
        rules="docs/510300_SAVED_STATE_FAILURE_DIAGNOSTIC_NEXT_20260909.md", goal_achieved=False, position_impact=0)
    paths = [Path(__file__), old_path, ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
        ROOT / "config/510300_research_authority_v6.json", SOURCE / "saved_verification_receipt.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for model in sorted({model for pair in CONTRASTS.values() for model in pair[:2]}):
                paths.append(SOURCE / period / cost / f"{model}_ledger.parquet")
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("保存状态诊断已固定48项比较，尚未分组或抽样。", flush=True)


def state_frame(data):
    wealth = data.wealth.to_numpy(float)
    returns = np.r_[np.nan, ((data.close+data.dividend)/data.previous_close-1).iloc[1:].to_numpy(float)]
    np.testing.assert_allclose(returns[1:], data.total_simple.iloc[1:], atol=1e-12, rtol=1e-12)
    trend = data.wealth/data.wealth.rolling(120, min_periods=120).mean()-1
    short = pd.Series(returns).rolling(20, min_periods=20).std(ddof=1)
    long = pd.Series(returns).rolling(60, min_periods=60).std(ddof=1)
    states = np.where(trend.notna() & short.notna() & long.notna(), 2*trend.gt(0).astype(int)+short.gt(long).astype(int), np.nan)
    # 用逐窗口标量计算检查全部可用分类，避免因列名或滞后误用形成假状态。
    for t in np.flatnonzero(np.isfinite(states)):
        mean_price = math.fsum(wealth[t-119:t+1])/120
        sigmas = []
        for window in [20, 60]:
            samples = returns[t-window+1:t+1]
            average = math.fsum(samples)/window
            sigmas.append(math.sqrt(math.fsum((value-average)**2 for value in samples)/(window-1)))
        require(int(states[t]) == 2*int(wealth[t] > mean_price)+int(sigmas[0] > sigmas[1]), "事前状态逐窗分类不一致")
    return pd.DataFrame({"date": data.date, "trend120": trend, "daily_vol20": short, "daily_vol60": long, "state": states})


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "保存状态诊断来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    states = state_frame(data)
    states.to_parquet(OUT / "market_states.parquet", index=False)
    rows, daily, matrices, whole = [], [], [], []
    for period, end, start in [("evaluation", cfg["data_cutoff"], cfg["evaluation_start"]),
        ("earlier_diagnostic", cfg["earlier_terminal"], cfg["earlier_start"])]:
        frame = data[data.date.le(end)]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        prior_state = states.state.iloc[indices].to_numpy(float)
        require(np.isfinite(prior_state).all(), "事前状态缺失，不以零组或删除行继续")
        columns = []
        for cost in cfg["costs"]:
            accounts = {}
            for model in sorted({model for pair in CONTRASTS.values() for model in pair[:2]}):
                ledger = pd.read_parquet(SOURCE / period / cost / f"{model}_ledger.parquet")
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "状态诊断账户日历不同")
                require(ledger.mark_clock.iloc[:-1].eq("CLOSE").all() and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL", "状态诊断终点时钟不同")
                require(pd.DatetimeIndex(ledger.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), "收益归组不是前一收盘状态")
                prior = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
                rates = np.column_stack([ledger.net_return, ledger.price_pnl/prior, ledger.dividend_recognized/prior,
                    (ledger.commission+ledger.slippage_cost)/prior])
                require(np.isfinite(rates).all(), "状态诊断收益缺失，不补零或删日")
                np.testing.assert_allclose(rates[:, 0], ledger.equity/prior-1, atol=1e-12, rtol=0)
                np.testing.assert_allclose(rates[:, 0], rates[:, 1]+rates[:, 2]-rates[:, 3], atol=1e-12, rtol=0)
                accounts[model] = rates
            for contrast, (left, right, label) in CONTRASTS.items():
                difference = accounts[left]-accounts[right]
                full = cfg["annual_days"]*difference.mean(axis=0)
                whole.append({"period": period, "cost": cost, "contrast": contrast, "annual_mean_net_difference": float(full[0]),
                    "annual_price_difference": float(full[1]), "annual_dividend_difference": float(full[2]), "annual_fee_difference": float(full[3])})
                summed = np.zeros(4)
                for state in STATES:
                    mask = prior_state == state
                    masked = difference*mask[:, None]
                    contribution = cfg["annual_days"]*masked.mean(axis=0)
                    summed += contribution
                    require(abs(contribution[0]-contribution[1]-contribution[2]+contribution[3]) < 1e-11, "状态损益分解不一致")
                    rows.append({"period": period, "cost": cost, "contrast": contrast, "comparison_name": label, "state": state, "state_name": STATES[state],
                        "observations": int(mask.sum()), "full_calendar_rows": len(mask), "fraction": float(mask.mean()),
                        "conditional_mean_net_difference_bp": float(difference[mask, 0].mean()*10000) if mask.any() else None,
                        "annual_mean_net_contribution": float(contribution[0]), "annual_price_contribution": float(contribution[1]),
                        "annual_dividend_contribution": float(contribution[2]), "annual_fee_contribution": float(contribution[3])})
                    columns.append(masked[:, 0])
                np.testing.assert_allclose(summed, full, atol=1e-11, rtol=0)
                daily.append(pd.DataFrame({"period": period, "cost": cost, "contrast": contrast, "origin": frame.date.iloc[indices].to_numpy(),
                    "date": frame.date.iloc[first:].to_numpy(), "state": prior_state.astype(int), "net_difference": difference[:, 0],
                    "price_difference": difference[:, 1], "dividend_difference": difference[:, 2], "fee_difference": difference[:, 3]}))
        matrices.append(np.column_stack(columns))
    table = pd.DataFrame(rows)
    require(len(table) == cfg["simultaneous_comparisons"] == 48, "保存状态诊断不等于48项")
    observed = table.annual_mean_net_contribution.to_numpy(float)
    bootstrap_checks = []
    for block in cfg["blocks"]:
        rng = np.random.default_rng(cfg["random_seed"]+block)
        samples = np.empty((cfg["repetitions"], len(table)))
        offset = 0
        for matrix in matrices:
            n, k = matrix.shape
            for first_sample in range(0, cfg["repetitions"], 100):
                size = min(100, cfg["repetitions"]-first_sample)
                starts = rng.integers(0, n, size=(size, math.ceil(n/block)))
                indices = ((starts[:, :, None]+np.arange(block)) % n).reshape(size, -1)[:, :n]
                samples[first_sample:first_sample+size, offset:offset+k] = cfg["annual_days"]*matrix[indices].mean(axis=1)
            offset += k
        standard_error = samples.std(axis=0, ddof=1)
        active = standard_error > 0
        maxima = np.max(abs((samples[:, active]-observed[active])/standard_error[active]), axis=1) if active.any() else np.zeros(cfg["repetitions"])
        critical = float(np.quantile(maxima, .95))
        table[f"block{block}_simultaneous_lower"] = observed-critical*standard_error
        table[f"block{block}_simultaneous_upper"] = observed+critical*standard_error
        table[f"block{block}_bootstrap_standard_error"] = standard_error
        bootstrap_checks.append({"block": block, "repetitions": cfg["repetitions"], "simultaneous_cells": len(table),
            "maximum_standardized_deviation_critical95": critical, "zero_bootstrap_variance_cells": int((~active).sum()), "same_period_paired_across_costs_and_comparisons": True})
        pd.DataFrame(samples, columns=[str(i) for i in range(len(table))]).to_parquet(OUT / f"bootstrap_contributions_block{block}.parquet", index=False)
        print(f"{block}日共同区块：48项同时区间、{cfg['repetitions']}次抽样完成。", flush=True)
    patterns = []
    for (contrast, state), group in table.groupby(["contrast", "state"]):
        require(len(group) == 4, "状态跨时期费用比较不完整")
        patterns.append({"contrast": contrast, "state": int(state), "state_name": STATES[int(state)],
            "all_four_point_contributions_positive": bool(group.annual_mean_net_contribution.gt(0).all()),
            "all_four_point_contributions_negative": bool(group.annual_mean_net_contribution.lt(0).all()),
            "all_four_both_blocks_lower_positive": bool(all(group[f"block{block}_simultaneous_lower"].gt(0).all() for block in cfg["blocks"])),
            "all_four_both_blocks_upper_negative": bool(all(group[f"block{block}_simultaneous_upper"].lt(0).all() for block in cfg["blocks"]))})
    table.to_csv(OUT / "state_contributions_and_simultaneous_intervals.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily, ignore_index=True).to_parquet(OUT / "paired_daily_attributions.parquet", index=False)
    pd.DataFrame(whole).to_csv(OUT / "full_calendar_reconciliation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(patterns).to_csv(OUT / "cross_period_state_patterns.csv", index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPLETED_DESCRIPTIVE_STATE_ATTRIBUTION_NO_NEW_STRATEGY_NO_PROMOTION",
        "comparison_cells": len(table), "paired_daily_rows": sum(len(frame) for frame in daily), "bootstrap_checks": bootstrap_checks,
        "cross_period_patterns": patterns, "new_accounts": 0, "new_model_fits": 0, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
        "run_seconds": time.perf_counter()-began, "source_and_full_calendar_reconciliation_checked": True, "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
