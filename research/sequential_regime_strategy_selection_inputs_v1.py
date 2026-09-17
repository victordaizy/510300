"""第179轮：在固定市场状态内，按当时已实现账户收益选择保存策略。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames


PRIMARY = "SEQUENTIAL_REGIME_STRATEGY_SELECTION"
MODELS = ["TREND_NOISE_REFERENCE_BLEND", "RUNS_OPPORTUNITY_CAPPED_SUM"]
CANDIDATES = {PRIMARY: "按状态历史实际收益顺序选择核心或相加封顶"}
ROOT = Path(__file__).resolve().parents[1]
PARENT_LEDGERS = {
    MODELS[0]: ROOT / "reports/research/510300_trend_noise_reference_blend_v1",
    MODELS[1]: ROOT / "reports/research/510300_runs_opportunity_union_v1",
}


def market_states(frame):
    wealth = frame.wealth.to_numpy(float)
    sma120 = np.full(len(frame), np.nan)
    peak60 = np.full(len(frame), np.nan)
    for index in range(119, len(frame)):
        sma120[index] = np.nanmean(wealth[index - 119:index + 1])
    for index in range(59, len(frame)):
        peak60[index] = np.nanmax(wealth[index - 59:index + 1])
    drawdown60 = wealth / peak60 - 1
    volatility20 = frame.vol20.to_numpy(float)
    state = np.full(len(frame), "未知", dtype=object)
    known = np.isfinite(sma120) & np.isfinite(drawdown60) & np.isfinite(volatility20)
    state[known & (drawdown60 <= -0.05)] = "压力回撤"
    state[known & (drawdown60 > -0.05) & (wealth > sma120) & (volatility20 > 0.20)] = "上升高波动"
    state[known & (drawdown60 > -0.05) & (wealth > sma120) & (volatility20 <= 0.20)] = "上升稳定"
    state[known & (drawdown60 > -0.05) & (wealth <= sma120)] = "非上升"
    return wealth, sma120, drawdown60, volatility20, state


def sequential_choice(state, candidate_returns, annual_days, lookback, minimum_samples, minimum_sharpe, minimum_annual_return):
    """对每个当日状态，只用该日及以前同状态的已实现账户净收益选策略；不合格则现金。"""
    names = list(candidate_returns)
    size = len(state)
    choice = np.full(size, "现金", dtype=object)
    score = np.full(size, np.nan)
    annual_return = np.full(size, np.nan)
    samples = np.zeros(size, dtype=int)
    for current in range(size):
        current_state = state[current]
        if current_state == "未知":
            continue
        left = max(0, current - lookback + 1)
        eligible = np.flatnonzero(state[left:current + 1] == current_state) + left
        if len(eligible) < minimum_samples:
            continue
        per_name = []
        for name in names:
            values = np.asarray(candidate_returns[name], dtype=float)[eligible]
            if not np.isfinite(values).all():
                continue
            average = float(values.mean() * annual_days)
            volatility = float(values.std(ddof=1) * np.sqrt(annual_days))
            sharpe = average / volatility if volatility > 0 else np.nan
            per_name.append((sharpe, average, name))
        qualified = [item for item in per_name if np.isfinite(item[0]) and item[0] >= minimum_sharpe and item[1] >= minimum_annual_return]
        if qualified:
            best = max(qualified, key=lambda item: (item[0], item[1], item[2]))
            score[current], annual_return[current], choice[current] = best
            samples[current] = len(eligible)
    return choice, score, annual_return, samples


def sequential_regime_strategy_selection_frames(data, parents_by_cost, cfg, start):
    require(cfg["candidate_models"] == list(CANDIDATES), "候选集合改变")
    require(cfg["parent_models"] == MODELS, "保存策略顺序改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    wealth, sma120, drawdown60, volatility20, state = market_states(data)
    origins = np.arange(first - 1, len(data) - 1)
    period = "evaluation" if str(start) == str(cfg["evaluation_start"]) else "earlier_diagnostic"
    output = []
    for cost, factors in frames.items():
        returns = {}
        for name in MODELS:
            ledger = pd.read_parquet(PARENT_LEDGERS[name] / period / cost / f"{name}_ledger.parquet")
            expected_dates = pd.DatetimeIndex(data.date[data.date.ge(start)])
            require(pd.DatetimeIndex(ledger.date).equals(expected_dates), "保存账户日历不同")
            aligned = np.full(len(data), np.nan)
            aligned[first:] = ledger.net_return.to_numpy(float)
            returns[name] = aligned
        choice, score, annual, samples = sequential_choice(
            state, returns, cfg["annual_days"], cfg["selection_lookback"], cfg["minimum_state_samples"],
            cfg["minimum_candidate_sharpe"], cfg["minimum_candidate_annual_return"],
        )
        target = np.full(len(data), np.nan)
        core_target = factors[MODELS[0] + "_parent_target"].to_numpy(float)
        capped_target = factors[MODELS[1] + "_parent_target"].to_numpy(float)
        core = (choice == MODELS[0]) & np.isfinite(core_target)
        capped = (choice == MODELS[1]) & np.isfinite(capped_target)
        target[core] = core_target[core]
        target[capped] = capped_target[capped]
        cash = (choice == "现金") & np.isfinite(core_target)
        target[cash] = 0.0
        factors["market_wealth"] = wealth
        factors["wealth_sma120"] = sma120
        factors["market_drawdown60"] = drawdown60
        factors["volatility20"] = volatility20
        factors["market_state"] = state
        factors["selected_strategy"] = choice
        factors["selected_state_sharpe"] = score
        factors["selected_state_annual_return"] = annual
        factors["selected_state_samples"] = samples
        factors[PRIMARY + "_target"] = target
        selected = choice[origins]
        output.append({
            "model": PRIMARY, "cost": cost, "decision_origins": len(origins),
            "core_origins": int((selected == MODELS[0]).sum()),
            "capped_sum_origins": int((selected == MODELS[1]).sum()),
            "cash_origins": int((selected == "现金").sum()),
            "unknown_target_origins": int(np.isnan(target[origins]).sum()),
            "mean_target": float(np.nanmean(target[origins])),
        })
    return frames, output
