"""执行冻结的R6_POSTMORTEM：暴露归因、静态零模型、事件、切分与安慰剂审计。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_r6_walkforward import (
    MATRIX_FILE,
    R5_CONFIG_FILE,
    _build_r5_positions,
    _candidate_targets,
    _costs,
    _grid_targets,
    _load_inputs,
    _run_variant,
)


PROTOCOL_FILE = ROOT / "config" / "r6_postmortem_protocol.yaml"
MODEL_SELECTION_FILE = ROOT / "reports" / "backtest" / "r6_model_selection.json"
OUTPUT_JSON = ROOT / "reports" / "backtest" / "r6_postmortem.json"
ATTRIBUTION_REPORT = ROOT / "reports" / "backtest" / "r6_exposure_attribution.md"
MATCHED_REPORT = ROOT / "reports" / "backtest" / "r6_matched_exposure_benchmarks.md"
EVENT_REPORT = ROOT / "reports" / "backtest" / "r6_signal_event_audit.md"
EVENT_CSV = ROOT / "reports" / "backtest" / "r6_signal_event_audit.csv"
SPLIT_REPORT = ROOT / "reports" / "backtest" / "r6_split_inversion_audit.md"
PLACEBO_REPORT = ROOT / "reports" / "backtest" / "r6_placebo_tests.md"
REJECTION_DECISION = ROOT / "docs" / "R6_REJECTION_DECISION.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_etf_total_return_series(
    etf: pd.DataFrame,
    dividends: pd.DataFrame,
    cash_annual_rate: float,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """构造510300含现金分红收益、现金收益及ETF相对现金收益。"""

    market = etf[["date", "open", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    cash_dividend = events.groupby("ex_date")["cash_dividend_per_share"].sum()
    market["cash_dividend_per_share"] = market["date"].map(cash_dividend).fillna(0.0)
    previous_close = market["close"].shift(1)
    market["etf_total_return"] = (
        (market["close"] + market["cash_dividend_per_share"]) / previous_close - 1.0
    ).fillna(0.0)
    market["cash_return"] = float(cash_annual_rate) / int(trading_days_per_year)
    market.loc[market.index[0], "cash_return"] = 0.0
    market["etf_minus_cash"] = market["etf_total_return"] - market["cash_return"]
    return market


def attribution_frame(
    ledger: pd.DataFrame,
    market_returns: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按冻结公式分解静态低暴露、动态择时、成本和复利残差。"""

    left = ledger.copy()
    left["date"] = pd.to_datetime(left["date"])
    data = left.merge(market_returns, on="date", how="left", validate="one_to_one")
    data = data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if data.empty or data[["etf_total_return", "cash_return"]].isna().any().any():
        raise ValueError("归因区间为空或基准收益缺失")
    weight = data["open_actual_position_after_trade"].astype(float).clip(0.0, 1.0)
    mean_weight = float(weight.mean())
    spread = data["etf_minus_cash"].astype(float)
    data["static_low_exposure_effect"] = -(1.0 - mean_weight) * spread
    data["dynamic_timing_contribution"] = (weight - mean_weight) * spread
    denominator = data["open_equity_before_trade"].replace(0.0, np.nan)
    data["execution_cost_return"] = (
        data["daily_total_execution_cost_cny"] / denominator
    ).fillna(0.0)
    data["timing_contribution_after_cost"] = (
        data["dynamic_timing_contribution"] - data["execution_cost_return"]
    )
    data["arithmetic_excess_attribution"] = (
        data["static_low_exposure_effect"]
        + data["dynamic_timing_contribution"]
        - data["execution_cost_return"]
    )
    strategy_total = float((1.0 + data["daily_return"]).prod() - 1.0)
    etf_total = float((1.0 + data["etf_total_return"]).prod() - 1.0)
    geometric_excess = strategy_total - etf_total
    arithmetic_total = float(data["arithmetic_excess_attribution"].sum())
    yearly = data.groupby(data["date"].dt.year).agg(
        static_low_exposure_effect=("static_low_exposure_effect", "sum"),
        dynamic_timing_contribution=("dynamic_timing_contribution", "sum"),
        execution_cost_return=("execution_cost_return", "sum"),
        timing_contribution_after_cost=("timing_contribution_after_cost", "sum"),
    )
    positive_timing = yearly["timing_contribution_after_cost"].clip(lower=0.0)
    year_concentration = (
        float(positive_timing.max() / positive_timing.sum())
        if positive_timing.sum() > 0.0
        else math.nan
    )
    summary = {
        "start_date": str(data["date"].min().date()),
        "end_date": str(data["date"].max().date()),
        "observations": int(len(data)),
        "mean_exposure": mean_weight,
        "static_low_exposure_effect": float(data["static_low_exposure_effect"].sum()),
        "dynamic_timing_contribution": float(data["dynamic_timing_contribution"].sum()),
        "execution_cost_return": float(data["execution_cost_return"].sum()),
        "timing_contribution_after_cost": float(
            data["timing_contribution_after_cost"].sum()
        ),
        "timing_contribution_after_cost_annualized_arithmetic": float(
            data["timing_contribution_after_cost"].mean() * 242
        ),
        "strategy_total_return": strategy_total,
        "etf_total_return": etf_total,
        "geometric_excess": geometric_excess,
        "arithmetic_attribution_total": arithmetic_total,
        "compounding_and_execution_residual": geometric_excess - arithmetic_total,
        "single_year_positive_timing_concentration": year_concentration,
        "yearly": {
            str(int(year)): {key: float(value) for key, value in row.items()}
            for year, row in yearly.iterrows()
        },
    }
    return data, summary


def _return_statistics(returns: pd.Series) -> dict[str, float]:
    values = returns.dropna().astype(float)
    years = len(values) / 242.0
    total = float((1.0 + values).prod() - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 and years > 0 else -1.0
    volatility = float(values.std(ddof=1) * math.sqrt(242))
    sharpe = float(values.mean() * 242 / volatility) if volatility > 0 else math.nan
    return {"total_return": total, "cagr": cagr, "volatility": volatility, "sharpe": sharpe}


def _relative_statistics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    difference = aligned["strategy"] - aligned["benchmark"]
    tracking_error = float(difference.std(ddof=1) * math.sqrt(242))
    information_ratio = (
        float(difference.mean() * 242 / tracking_error) if tracking_error > 0 else math.nan
    )
    return {
        "cagr_excess": _return_statistics(aligned["strategy"])["cagr"]
        - _return_statistics(aligned["benchmark"])["cagr"],
        "information_ratio": information_ratio,
    }


def matched_static_benchmarks(data: pd.DataFrame) -> dict[str, Any]:
    """生成同平均仓位和同波动静态组合，并计算低Beta之外的增量。"""

    strategy = data.set_index("date")["daily_return"].astype(float)
    etf = data.set_index("date")["etf_total_return"].astype(float)
    cash = data.set_index("date")["cash_return"].astype(float)
    mean_weight = float(data["open_actual_position_after_trade"].mean())
    static_mean = mean_weight * etf + (1.0 - mean_weight) * cash
    strategy_vol = float(strategy.std(ddof=1) * math.sqrt(242))
    etf_excess_vol = float((etf - cash).std(ddof=1) * math.sqrt(242))
    vol_matched_weight = min(max(strategy_vol / etf_excess_vol, 0.0), 1.0) if etf_excess_vol > 0 else 0.0
    static_vol = vol_matched_weight * etf + (1.0 - vol_matched_weight) * cash
    x = (etf - cash).to_numpy()
    y = (strategy - cash).to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    alpha_daily, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    positive = etf.gt(cash)
    negative = etf.lt(cash)
    upside_capture = (
        float((strategy[positive] - cash[positive]).mean() / (etf[positive] - cash[positive]).mean())
        if positive.any() and (etf[positive] - cash[positive]).mean() != 0
        else math.nan
    )
    downside_capture = (
        float((strategy[negative] - cash[negative]).mean() / (etf[negative] - cash[negative]).mean())
        if negative.any() and (etf[negative] - cash[negative]).mean() != 0
        else math.nan
    )
    return {
        "strategy": _return_statistics(strategy),
        "static_mean_exposure": {
            "weight": mean_weight,
            **_return_statistics(static_mean),
            **_relative_statistics(strategy, static_mean),
        },
        "static_vol_matched": {
            "weight": vol_matched_weight,
            **_return_statistics(static_vol),
            **_relative_statistics(strategy, static_vol),
        },
        "regression_alpha_annualized": float(alpha_daily * 242),
        "regression_beta": float(beta),
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
    }


def circular_shift_placebo(
    weight: pd.Series,
    spread: pd.Series,
    cost: pd.Series,
    *,
    repetitions: int,
    minimum_shift: int,
    seed: int,
) -> dict[str, Any]:
    """循环平移仓位路径，保留仓位分布和持续性。"""

    w = weight.to_numpy(dtype=float)
    r = spread.to_numpy(dtype=float)
    c = cost.to_numpy(dtype=float)
    if len(w) <= 2 * minimum_shift + 1:
        raise ValueError("循环平移区间过短")
    centered = w - w.mean()
    valid_shifts = np.arange(minimum_shift, len(w) - minimum_shift + 1)
    all_values = np.array(
        [float(np.mean(np.roll(centered, int(shift)) * r - c)) for shift in valid_shifts]
    ) * 242.0
    rng = np.random.default_rng(seed)
    sampled = rng.choice(all_values, size=repetitions, replace=True)
    actual = float(np.mean(centered * r - c) * 242.0)
    percentile = float(np.mean(sampled <= actual))
    return {
        "actual_timing_after_cost_annualized": actual,
        "repetitions": repetitions,
        "minimum_shift": minimum_shift,
        "valid_unique_shifts": int(len(valid_shifts)),
        "placebo_percentile": percentile,
        "one_sided_p_value": 1.0 - percentile,
        "placebo_quantiles": {
            "p05": float(np.quantile(sampled, 0.05)),
            "p50": float(np.quantile(sampled, 0.50)),
            "p95": float(np.quantile(sampled, 0.95)),
        },
    }


def _block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = math.ceil(length / block_length)
    starts = rng.integers(0, max(length - block_length + 1, 1), size=blocks)
    indices = np.concatenate(
        [np.arange(start, min(start + block_length, length)) for start in starts]
    )
    while len(indices) < length:
        start = int(rng.integers(0, max(length - block_length + 1, 1)))
        indices = np.concatenate([indices, np.arange(start, min(start + block_length, length))])
    return indices[:length]


def block_bootstrap(
    data: pd.DataFrame,
    *,
    repetitions: int,
    block_length: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """对成对的仓位和收益路径做移动分块Bootstrap。"""

    timing = data["timing_contribution_after_cost"].to_numpy(dtype=float)
    strategy = data["daily_return"].to_numpy(dtype=float)
    static_mean = (
        float(data["open_actual_position_after_trade"].mean())
        * data["etf_total_return"].to_numpy(dtype=float)
        + (1.0 - float(data["open_actual_position_after_trade"].mean()))
        * data["cash_return"].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    timing_values = np.empty(repetitions)
    static_excess_values = np.empty(repetitions)
    sharpe_values = np.empty(repetitions)
    for iteration in range(repetitions):
        indices = _block_indices(len(data), block_length, rng)
        timing_values[iteration] = timing[indices].mean() * 242.0
        static_excess_values[iteration] = (
            strategy[indices].mean() - static_mean[indices].mean()
        ) * 242.0
        volatility = strategy[indices].std(ddof=1) * math.sqrt(242)
        sharpe_values[iteration] = (
            strategy[indices].mean() * 242.0 / volatility if volatility > 0 else math.nan
        )
    tail = (1.0 - confidence_level) / 2.0

    def interval(values: np.ndarray) -> dict[str, float]:
        clean = values[np.isfinite(values)]
        return {
            "median": float(np.quantile(clean, 0.50)),
            "lower": float(np.quantile(clean, tail)),
            "upper": float(np.quantile(clean, 1.0 - tail)),
        }

    return {
        "repetitions": repetitions,
        "block_length": block_length,
        "confidence_level": confidence_level,
        "timing_after_cost_annualized": interval(timing_values),
        "excess_vs_static_mean_annualized_arithmetic": interval(static_excess_values),
        "strategy_sharpe": interval(sharpe_values),
    }


def _candidate_seed(base_seed: int, candidate_id: str, suffix: int = 0) -> int:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16) + suffix) % (2**32 - 1)


def build_event_audit(
    candidate_id: str,
    family: str,
    variant: str,
    targets: pd.DataFrame,
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    market_returns: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """为每一次冻结目标仓位变化建立后续收益和成本记录。"""

    signals = targets.loc[targets["target_position"].notna()].copy()
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.sort_values("date").reset_index(drop=True)
    signals["old_weight"] = signals["target_position"].shift(1)
    changes = signals.loc[
        signals["old_weight"].notna()
        & signals["target_position"].sub(signals["old_weight"]).abs().gt(1e-12)
    ].copy()
    calendar = pd.DatetimeIndex(market_returns["date"])
    market_by_date = market_returns.set_index("date")
    costs = ledger.set_index("date")[
        ["daily_total_execution_cost_cny", "open_equity_before_trade"]
    ]
    trade_costs = (
        trades.assign(
            total_cost_cny=(trades["commission"] + trades["stamp_duty"])
            + (trades["execution_price"] - trades["open_price"]).abs() * trades["quantity"]
        ).groupby("date")["total_cost_cny"].sum()
        if not trades.empty
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for row in changes.itertuples(index=False):
        signal_date = pd.Timestamp(row.date)
        position = int(calendar.searchsorted(signal_date, side="right"))
        if position >= len(calendar):
            continue
        execution_date = pd.Timestamp(calendar[position])
        delta = float(row.target_position - row.old_weight)
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "family": family,
            "variant": variant,
            "signal_date": signal_date,
            "execution_date": execution_date,
            "old_weight": float(row.old_weight),
            "new_weight": float(row.target_position),
            "weight_change": delta,
            "direction": "INCREASE" if delta > 0 else "DECREASE",
            "reason": str(row.signal_reason),
            "cost_cny": float(trade_costs.get(execution_date, 0.0)),
            "signal_close_price": float(market_by_date.loc[signal_date, "close"]),
            "execution_open_price": float(market_by_date.loc[execution_date, "open"]),
        }
        if execution_date in costs.index:
            cost_row = costs.loc[execution_date]
            record["cost_return"] = float(
                cost_row["daily_total_execution_cost_cny"]
                / cost_row["open_equity_before_trade"]
            ) if cost_row["open_equity_before_trade"] else 0.0
        else:
            record["cost_return"] = 0.0
        for horizon in horizons:
            future_dates = calendar[position : position + horizon]
            if len(future_dates) < horizon:
                record[f"next_{horizon}d_end_date"] = pd.NaT
                record[f"next_{horizon}d_end_close"] = math.nan
                record[f"next_{horizon}d_etf_total_return"] = math.nan
                record[f"next_{horizon}d_cash_return"] = math.nan
                record[f"next_{horizon}d_etf_minus_cash"] = math.nan
                record[f"timing_gain_{horizon}d"] = math.nan
                continue
            window = market_by_date.reindex(future_dates)
            end_date = pd.Timestamp(future_dates[-1])
            end_close = float(window.iloc[-1]["close"])
            execution_open = float(record["execution_open_price"])
            dividends = float(window["cash_dividend_per_share"].sum())
            etf_forward_return = (end_close + dividends) / execution_open - 1.0
            cash_forward_return = float((1.0 + window["cash_return"]).prod() - 1.0)
            forward = etf_forward_return - cash_forward_return
            record[f"next_{horizon}d_end_date"] = end_date
            record[f"next_{horizon}d_end_close"] = end_close
            record[f"next_{horizon}d_etf_total_return"] = etf_forward_return
            record[f"next_{horizon}d_cash_return"] = cash_forward_return
            record[f"next_{horizon}d_etf_minus_cash"] = forward
            record[f"timing_gain_{horizon}d"] = delta * forward
        forward20 = record["next_20d_etf_minus_cash"]
        if not math.isfinite(float(forward20)):
            decision = "PENDING_FORWARD_WINDOW"
        elif delta < 0 and forward20 < 0:
            decision = "CORRECT_REDUCTION"
        elif delta < 0 and forward20 >= 0:
            decision = "WRONG_REDUCTION_MISSED_UPSIDE"
        elif delta > 0 and forward20 >= 0:
            decision = "CORRECT_INCREASE"
        else:
            decision = "WRONG_INCREASE"
        record["decision_20d"] = decision
        rows.append(record)
    return pd.DataFrame(rows)


def _event_summary(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"event_count": 0}
    completed = events.loc[events["timing_gain_20d"].notna()].copy()
    if completed.empty:
        return {
            "event_count": int(len(events)),
            "completed_20d_event_count": 0,
            "pending_20d_event_count": int(len(events)),
        }
    contribution = completed["timing_gain_20d"] - completed["cost_return"]
    positive = contribution.clip(lower=0.0)
    event_concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else math.nan
    by_year = contribution.groupby(pd.to_datetime(completed["signal_date"]).dt.year).sum()
    positive_year = by_year.clip(lower=0.0)
    year_concentration = (
        float(positive_year.max() / positive_year.sum()) if positive_year.sum() > 0 else math.nan
    )
    decision = completed.assign(net_timing_gain_20d=contribution).groupby("decision_20d").agg(
        count=("decision_20d", "size"),
        net_timing_gain_20d=("net_timing_gain_20d", "sum"),
    )
    return {
        "event_count": int(len(events)),
        "completed_20d_event_count": int(len(completed)),
        "pending_20d_event_count": int(len(events) - len(completed)),
        "decision_counts": {
            name: {
                "count": int(row["count"]),
                "net_timing_gain_20d": float(row["net_timing_gain_20d"]),
            }
            for name, row in decision.iterrows()
        },
        "single_event_positive_concentration": event_concentration,
        "single_year_positive_event_concentration": year_concentration,
    }


def _rank_stability(model_selection: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in model_selection["ranking"]:
        rows.append(
            {
                "id": item["id"],
                "family": item["family"],
                "development": item["continuous_periods"]["development"].get("sharpe"),
                "validation": item["continuous_periods"]["validation"].get("sharpe"),
                "historical_pseudo_oos": item["continuous_periods"][
                    "historical_pseudo_oos"
                ].get("sharpe"),
            }
        )
    frame = pd.DataFrame(rows).set_index("id")
    ranks = frame[["development", "validation", "historical_pseudo_oos"]].rank(
        ascending=False, method="min"
    )
    development_validation = spearmanr(
        frame["development"], frame["validation"], nan_policy="omit"
    )
    validation_pseudo = spearmanr(
        frame["validation"], frame["historical_pseudo_oos"], nan_policy="omit"
    )
    top = {
        column: set(frame[column].nlargest(5).index)
        for column in ("development", "validation", "historical_pseudo_oos")
    }
    detail = frame.join(ranks, rsuffix="_rank").reset_index()
    detail["validation_to_pseudo_rank_change"] = (
        detail["historical_pseudo_oos_rank"] - detail["validation_rank"]
    )
    family = detail.groupby("family").agg(
        development_sharpe=("development", "mean"),
        validation_sharpe=("validation", "mean"),
        pseudo_oos_sharpe=("historical_pseudo_oos", "mean"),
        mean_absolute_rank_change=("validation_to_pseudo_rank_change", lambda values: values.abs().mean()),
    )
    return {
        "development_validation_spearman": float(development_validation.statistic),
        "development_validation_p_value": float(development_validation.pvalue),
        "validation_pseudo_oos_spearman": float(validation_pseudo.statistic),
        "validation_pseudo_oos_p_value": float(validation_pseudo.pvalue),
        "top5_overlap_development_validation": float(
            len(top["development"] & top["validation"]) / 5.0
        ),
        "top5_overlap_validation_pseudo_oos": float(
            len(top["validation"] & top["historical_pseudo_oos"]) / 5.0
        ),
        "top5": {name: sorted(values) for name, values in top.items()},
        "candidate_rank_changes": detail.to_dict("records"),
        "family_rank_changes": {
            name: {key: float(value) for key, value in row.items()}
            for name, row in family.iterrows()
        },
    }


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value):.2%}"


def _render_attribution(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# R6低暴露—动态择时—成本归因",
        "",
        "> 主指标为验证期 `TIMING_CONTRIBUTION_AFTER_COST`。数值为区间累计算术贡献；复利残差单独保留。",
        "",
        "| 候选 | 族 | 验证平均仓位 | 静态低仓位 | 动态择时 | 成本 | 成本后择时 | 伪OOS成本后择时 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        validation = item["attribution"]["CONTINUOUS"]["validation"]
        pseudo = item["attribution"]["CONTINUOUS"]["historical_pseudo_oos"]
        lines.append(
            f"| {item['id']} | {item['family']} | {_fmt_pct(validation['mean_exposure'])} | "
            f"{_fmt_pct(validation['static_low_exposure_effect'])} | "
            f"{_fmt_pct(validation['dynamic_timing_contribution'])} | "
            f"{_fmt_pct(-validation['execution_cost_return'])} | "
            f"{_fmt_pct(validation['timing_contribution_after_cost'])} | "
            f"{_fmt_pct(pseudo['timing_contribution_after_cost'])} |"
        )
    lines.extend(
        [
            "",
            "连续、10%和25%三种路径均已计算并保存在机器可读报告中。静态低仓位效应与动态择时之和减成本后，与实际几何超额的差记为复利及执行残差。",
        ]
    )
    return "\n".join(lines)


def _render_matched(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# R6同平均仓位与同波动静态基准",
        "",
        "| 候选 | 验证平均仓位 | 对同均仓CAGR差 | 同波动仓位 | 对同波动CAGR差 | 年化回归Alpha | 上行捕获 | 下行捕获 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        matched = item["matched_benchmarks"]["validation"]
        lines.append(
            f"| {item['id']} | {_fmt_pct(matched['static_mean_exposure']['weight'])} | "
            f"{_fmt_pct(matched['static_mean_exposure']['cagr_excess'])} | "
            f"{_fmt_pct(matched['static_vol_matched']['weight'])} | "
            f"{_fmt_pct(matched['static_vol_matched']['cagr_excess'])} | "
            f"{_fmt_pct(matched['regression_alpha_annualized'])} | "
            f"{matched['upside_capture']:.3f} | {matched['downside_capture']:.3f} |"
        )
    return "\n".join(lines)


def _render_events(events: pd.DataFrame, leader_id: str, candidate_summaries: dict[str, Any]) -> str:
    leader = events.loc[
        events["candidate_id"].eq(leader_id) & events["variant"].eq("CONTINUOUS")
    ].copy()
    leader["net_timing_gain_20d"] = leader["timing_gain_20d"] - leader["cost_return"]
    positive = leader.nlargest(5, "net_timing_gain_20d")
    negative = leader.nsmallest(5, "net_timing_gain_20d")
    lines = [
        "# R6仓位变化事件级审计",
        "",
        f"完整逐事件记录见 `{EVENT_CSV.relative_to(ROOT).as_posix()}`。下表聚焦历史验证排序第一但未获批准的 `{leader_id}` 连续路径。",
        "信号在当日收盘确认，下一交易日开盘执行；5/20/60日收益从执行开盘计至完整窗口末日收盘并计入现金分红。样本尾部不完整窗口标为待观察，不用截短收益替代。",
        "",
        "## 决策分类",
        "",
        "| 分类 | 次数 | 20日净择时贡献 |",
        "|---|---:|---:|",
    ]
    summary = candidate_summaries[leader_id]
    for name, detail in summary.get("decision_counts", {}).items():
        lines.append(f"| {name} | {detail['count']} | {_fmt_pct(detail['net_timing_gain_20d'])} |")
    for title, frame in (("最大五次正贡献", positive), ("最大五次负贡献", negative)):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 信号日 | 信号收盘 | 执行日 | 执行开盘 | 方向 | 旧信号 | 新信号 | 20日截止日 | 后20日ETF减现金 | 净择时贡献 | 原因 |",
                "|---|---:|---|---:|---|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in frame.itertuples(index=False):
            lines.append(
                f"| {pd.Timestamp(row.signal_date).date()} | {row.signal_close_price:.3f} | "
                f"{pd.Timestamp(row.execution_date).date()} | {row.execution_open_price:.3f} | "
                f"{row.direction} | {_fmt_pct(row.old_weight)} | {_fmt_pct(row.new_weight)} | "
                f"{pd.Timestamp(row.next_20d_end_date).date()} | "
                f"{_fmt_pct(row.next_20d_etf_minus_cash)} | {_fmt_pct(row.net_timing_gain_20d)} | {row.reason} |"
            )
    return "\n".join(lines)


def _render_split(split: dict[str, Any], boundaries: list[dict[str, Any]], exposures: dict[str, Any]) -> str:
    lines = [
        "# R6验证期—历史伪OOS排序反转审计",
        "",
        f"- 开发—验证Spearman：{split['development_validation_spearman']:.3f}。",
        f"- 验证—历史伪OOS Spearman：{split['validation_pseudo_oos_spearman']:.3f}。",
        f"- 开发/验证Top 5重合率：{split['top5_overlap_development_validation']:.0%}。",
        f"- 验证/历史伪OOS Top 5重合率：{split['top5_overlap_validation_pseudo_oos']:.0%}。",
        "",
        "## 实际日期边界",
        "",
        "| 候选 | 原始预热起点 | 首个有效信号 | 首个可成交日 | 验证区间 | 伪OOS区间 | 最后成交日 | 最后估值日 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in boundaries:
        lines.append(
            f"| {item['id']} | {item['feature_warmup_start']} | {item['first_valid_signal_date']} | "
            f"{item['first_execution_date']} | {item['validation_period']} | "
            f"{item['historical_pseudo_oos_period']} | {item['last_trade_date']} | {item['last_valuation_date']} |"
        )
    lines.extend(
        [
            "",
            "## 分区间平均仓位（连续路径）",
            "",
            "| 候选 | 开发 | 验证 | 历史伪OOS |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate_id, values in exposures.items():
        lines.append(
            f"| {candidate_id} | {_fmt_pct(values['development'])} | {_fmt_pct(values['validation'])} | "
            f"{_fmt_pct(values['historical_pseudo_oos'])} |"
        )
    return "\n".join(lines)


def _render_placebo(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# R6循环平移与分块Bootstrap",
        "",
        "循环平移使用验证期连续仓位，重复5000次且至少错开20个交易日。百分位达到95%才算通过。",
        "",
        "| 候选 | 实际成本后择时年化 | 循环平移百分位 | 单侧p值 | 20日块择时95%区间 | 60日块择时95%区间 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in candidates:
        placebo = item["placebo"]
        b20 = item["bootstrap"]["20"]["timing_after_cost_annualized"]
        b60 = item["bootstrap"]["60"]["timing_after_cost_annualized"]
        lines.append(
            f"| {item['id']} | {_fmt_pct(placebo['actual_timing_after_cost_annualized'])} | "
            f"{placebo['placebo_percentile']:.1%} | {placebo['one_sided_p_value']:.3f} | "
            f"[{_fmt_pct(b20['lower'])}, {_fmt_pct(b20['upper'])}] | "
            f"[{_fmt_pct(b60['lower'])}, {_fmt_pct(b60['upper'])}] |"
        )
    return "\n".join(lines)


def _render_rejection(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    family = payload["family_summary"]
    lines = [
        "# R6历史拒绝决策",
        "",
        "状态：`HISTORICALLY_REJECTED`",
        "下一阶段：`R6_POSTMORTEM_COMPLETE`",
        "",
        "```text",
        "historical_candidates = 25",
        "historical_passed = 0",
        "historical_best_candidate = B_CONT_W756",
        "historical_best_is_approved = false",
        "r6_forward_1_enabled = false",
        "live_position_mapping_enabled = false",
        "order_generation_enabled = false",
        "new_factor_research_enabled = false",
        "machine_learning_research_enabled = false",
        "next_phase = R6_POSTMORTEM_COMPLETE",
        "```",
        "",
        "## 归因结论",
        "",
        f"- 通过全部失败归因继续研究门槛的候选：{decision['postmortem_gate_pass_count']}个。",
        f"- 验证—历史伪OOS Sharpe秩相关：{payload['split_inversion']['validation_pseudo_oos_spearman']:.3f}。",
        f"- 10%档与25%小账户超额秩相关：{payload['execution_mapping_rank_correlation']:.3f}。",
    ]
    for family_name, summary in family.items():
        lines.append(
            f"- {family_name}：验证期族中位成本后动态择时贡献"
            f"{_fmt_pct(summary['median_validation_timing_after_cost'])}，"
            f"循环平移中位百分位{summary['median_placebo_percentile']:.1%}。"
        )
    lines.extend(
        [
            "",
            "## 正式决定",
            "",
            decision["formal_conclusion"],
            "",
            "R5仅保留为历史基准和工程回归系统；R6 25候选冻结且不得追加；R6_FORWARD_1继续禁用；V3仅按原协议观察；PCF/IOPV继续采集但不参与仓位；分钟信息和MACD不恢复为Alpha；券商连接与订单生成保持禁用。",
            "",
            "本结论只说明当前单ETF择时变量族缺乏验证支持。若未来改变为风险控制目标，必须先通过同平均仓位和同波动静态基准；若仍追求长期正超额，需要另立协议讨论扩大投资范围，不能把它包装成R6参数优化。",
            "",
            "## 可选失败族观察档案",
            "",
            "`R6_FAILED_FAMILY_OBSERVATION` 从2026-08-17起只追加当日新数据，不回填缺失日期；主统计量固定为三族中位数，满242个新交易日之前禁止评价。它不显示候选排名、不批准候选、不映射仓位、不连接券商也不生成订单。",
            "",
            "运行入口：`.venv\\Scripts\\python.exe scripts\\run_r6_failed_family_observation.py`。若当日数据尚不可用，状态文件必须明确报告旧数据日期，且不得把旧结果记成今日观察。",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    protocol = yaml.safe_load(PROTOCOL_FILE.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX_FILE.read_text(encoding="utf-8"))
    if _sha256(MATRIX_FILE) != protocol["protocol"]["source_experiment_matrix_sha256"]:
        raise ValueError("R6实验矩阵哈希与归因协议冻结值不一致")
    if len(matrix["candidates"]) != int(protocol["protocol"]["historical_candidates"]):
        raise ValueError("R6候选数量与归因协议不一致")
    model_selection = json.loads(MODEL_SELECTION_FILE.read_text(encoding="utf-8"))
    frames, paths = _load_inputs(matrix)
    r5_config = yaml.safe_load(R5_CONFIG_FILE.read_text(encoding="utf-8"))
    r5_positions = _build_r5_positions(frames["index"], frames["valuation"], r5_config)
    execution = matrix["execution"]
    base_costs = _costs(execution)
    research_cash = float(execution["continuous_research_account_cny"])
    small_cash = float(execution["small_account_cny"])
    market_returns = build_etf_total_return_series(
        frames["etf"],
        frames["dividends"],
        float(protocol["attribution"]["cash_annual_rate"]),
        int(protocol["attribution"]["trading_days_per_year"]),
    )
    periods = protocol["periods"]
    variants = {
        "CONTINUOUS": {"step": None, "cash": research_cash, "minimum_shares": 0},
        "GRID_10PCT": {
            "step": float(execution["research_grid_step"]),
            "cash": research_cash,
            "minimum_shares": 0,
        },
        "SMALL_ACCOUNT_25PCT": {
            "step": float(execution["small_account_grid_step"]),
            "cash": small_cash,
            "minimum_shares": int(execution["small_account_minimum_normal_trade_shares"]),
        },
    }
    candidate_results: list[dict[str, Any]] = []
    all_events: list[pd.DataFrame] = []
    boundaries: list[dict[str, Any]] = []
    split_exposures: dict[str, Any] = {}
    candidate_event_summaries: dict[str, Any] = {}
    daily_cache: dict[str, dict[str, pd.DataFrame]] = {}

    for candidate in matrix["candidates"]:
        base_targets = _candidate_targets(candidate, frames["etf"], r5_positions, execution)
        result: dict[str, Any] = {
            "id": candidate["id"],
            "family": candidate["family"],
            "neighbor_group": candidate["neighbor_group"],
            "attribution": {},
            "matched_benchmarks": {},
        }
        daily_cache[candidate["id"]] = {}
        variant_ledgers: dict[str, pd.DataFrame] = {}
        variant_trades: dict[str, pd.DataFrame] = {}
        continuous_event_frame: pd.DataFrame | None = None
        for variant_name, definition in variants.items():
            targets = (
                base_targets
                if definition["step"] is None
                else _grid_targets(base_targets, float(definition["step"]))
            )
            ledger, trades = _run_variant(
                etf=frames["etf"],
                dividends=frames["dividends"],
                targets=targets,
                initial_cash=float(definition["cash"]),
                costs=base_costs,
                minimum_trade_shares=int(definition["minimum_shares"]),
            )
            variant_ledgers[variant_name] = ledger
            variant_trades[variant_name] = trades
            result["attribution"][variant_name] = {}
            for period_name in ("development", "validation", "historical_pseudo_oos"):
                start, end = periods[period_name]
                daily, summary = attribution_frame(ledger, market_returns, start, end)
                result["attribution"][variant_name][period_name] = summary
                if variant_name == "CONTINUOUS":
                    daily_cache[candidate["id"]][period_name] = daily
                    result["matched_benchmarks"][period_name] = matched_static_benchmarks(daily)
            event_frame = build_event_audit(
                candidate["id"],
                candidate["family"],
                variant_name,
                targets,
                ledger,
                trades,
                market_returns,
                [int(value) for value in protocol["attribution"]["event_forward_horizons"]],
            )
            all_events.append(event_frame)
            if variant_name == "CONTINUOUS":
                continuous_event_frame = event_frame

        validation_daily = daily_cache[candidate["id"]]["validation"]
        placebo_config = protocol["placebo"]
        base_seed = int(placebo_config["random_seed"])
        result["placebo"] = circular_shift_placebo(
            validation_daily["open_actual_position_after_trade"],
            validation_daily["etf_minus_cash"],
            validation_daily["execution_cost_return"],
            repetitions=int(placebo_config["circular_shift_repetitions"]),
            minimum_shift=int(placebo_config["minimum_circular_shift_trading_days"]),
            seed=_candidate_seed(base_seed, candidate["id"]),
        )
        result["bootstrap"] = {}
        for block_length in placebo_config["block_lengths_trading_days"]:
            result["bootstrap"][str(block_length)] = block_bootstrap(
                validation_daily,
                repetitions=int(placebo_config["block_bootstrap_repetitions"]),
                block_length=int(block_length),
                confidence_level=float(placebo_config["confidence_level"]),
                seed=_candidate_seed(base_seed, candidate["id"], int(block_length)),
            )

        subperiod_timing: dict[str, float] = {}
        for subperiod in periods["predefined_stability_subperiods"]:
            _, summary = attribution_frame(
                variant_ledgers["CONTINUOUS"],
                market_returns,
                subperiod["start"],
                subperiod["end"],
            )
            subperiod_timing[subperiod["id"]] = float(
                summary["timing_contribution_after_cost"]
            )
        result["predefined_subperiod_timing_after_cost"] = subperiod_timing

        if continuous_event_frame is None:
            raise RuntimeError(f"{candidate['id']}缺少连续路径事件审计")
        candidate_event_summaries[candidate["id"]] = _event_summary(
            continuous_event_frame
        )
        ready = base_targets.loc[base_targets["target_position"].notna()].copy()
        calendar = pd.DatetimeIndex(frames["etf"]["date"])
        first_signal = pd.Timestamp(ready["date"].min())
        first_execution_position = int(calendar.searchsorted(first_signal, side="right"))
        first_execution = (
            pd.Timestamp(calendar[first_execution_position])
            if first_execution_position < len(calendar)
            else pd.NaT
        )
        continuous_trades = variant_trades["CONTINUOUS"]
        boundaries.append(
            {
                "id": candidate["id"],
                "feature_warmup_start": str(pd.Timestamp(frames["etf"]["date"].min()).date()),
                "first_valid_signal_date": str(first_signal.date()),
                "first_execution_date": str(first_execution.date()),
                "development_period": " / ".join(periods["development"]),
                "validation_period": " / ".join(periods["validation"]),
                "historical_pseudo_oos_period": " / ".join(
                    periods["historical_pseudo_oos"]
                ),
                "last_trade_date": (
                    str(pd.Timestamp(continuous_trades["date"].max()).date())
                    if not continuous_trades.empty
                    else "NONE"
                ),
                "last_valuation_date": str(pd.Timestamp(frames["valuation"]["date"].max()).date()),
            }
        )
        split_exposures[candidate["id"]] = {
            period_name: float(
                daily_cache[candidate["id"]][period_name][
                    "open_actual_position_after_trade"
                ].mean()
            )
            for period_name in ("development", "validation", "historical_pseudo_oos")
        }
        candidate_results.append(result)

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events = events.sort_values(["candidate_id", "variant", "signal_date"]).reset_index(drop=True)
    EVENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENT_CSV, index=False, encoding="utf-8-sig")

    split_inversion = _rank_stability(model_selection)
    grid_excess = np.array(
        [
            item["periods"]["historical_pseudo_oos"]["annualized_excess_vs_etf"]
            for item in model_selection["ranking"]
        ]
    )
    small_excess = np.array(
        [
            item["small_account_historical_pseudo_oos"]["annualized_excess_vs_etf"]
            for item in model_selection["ranking"]
        ]
    )
    execution_rank_correlation = float(spearmanr(grid_excess, small_excess).statistic)

    result_by_id = {item["id"]: item for item in candidate_results}
    b_neighbor_consistency: dict[str, bool] = {}
    for mapping_prefix in ("B_MATRIX", "B_CONT", "B_TREND_LIGHT", "B_RISK_LIGHT"):
        ids = [candidate_id for candidate_id in result_by_id if candidate_id.startswith(mapping_prefix)]
        if len(ids) == 2:
            signs = [
                np.sign(
                    result_by_id[candidate_id]["attribution"]["CONTINUOUS"]["validation"][
                        "timing_contribution_after_cost"
                    ]
                )
                for candidate_id in ids
            ]
            consistent = bool(signs[0] == signs[1])
            for candidate_id in ids:
                b_neighbor_consistency[candidate_id] = consistent

    gates = protocol["continuation_gates"]
    for item in candidate_results:
        validation = item["attribution"]["CONTINUOUS"]["validation"]
        matched = item["matched_benchmarks"]["validation"]
        event_summary = candidate_event_summaries[item["id"]]
        variant_signs = [
            np.sign(
                item["attribution"][variant]["validation"][
                    "timing_contribution_after_cost"
                ]
            )
            for variant in variants
        ]
        selection_item = next(row for row in model_selection["ranking"] if row["id"] == item["id"])
        selection_metrics = selection_item["continuous_periods"]["validation"]
        primary_signs = [
            np.sign(selection_metrics["annualized_excess_vs_etf"]),
            np.sign(selection_metrics["annualized_excess_vs_h00300"]),
        ]
        concentration_values = [
            value
            for value in (
                validation["single_year_positive_timing_concentration"],
                event_summary.get("single_event_positive_concentration", math.nan),
                event_summary.get("single_year_positive_event_concentration", math.nan),
            )
            if math.isfinite(float(value))
        ]
        maximum_concentration = max(concentration_values) if concentration_values else math.inf
        checks = {
            "validation_timing_after_cost_positive": validation[
                "timing_contribution_after_cost"
            ]
            > float(gates["validation_timing_after_cost_minimum"]),
            "subperiod_stability": sum(value > 0.0 for value in item["predefined_subperiod_timing_after_cost"].values())
            >= int(gates["positive_predefined_subperiods_minimum"]),
            "beats_static_mean": matched["static_mean_exposure"]["cagr_excess"]
            > float(gates["excess_vs_static_mean_minimum"]),
            "beats_static_vol_matched": matched["static_vol_matched"]["cagr_excess"]
            > float(gates["excess_vs_static_vol_matched_minimum"]),
            "circular_shift_front_5pct": item["placebo"]["placebo_percentile"]
            >= float(gates["circular_shift_percentile_minimum"]),
            "concentration": maximum_concentration
            <= float(gates["maximum_single_year_or_event_concentration"]),
            "execution_variant_direction_consistency": len(set(variant_signs)) == 1,
            "parameter_neighbor_consistency": b_neighbor_consistency.get(item["id"], True),
            "primary_benchmark_direction_consistency": primary_signs[0] == primary_signs[1],
        }
        item["postmortem_gate_checks"] = checks
        item["postmortem_gate_pass"] = all(checks.values())
        item["event_summary"] = event_summary

    family_summary: dict[str, Any] = {}
    for family in sorted({item["family"] for item in candidate_results}):
        members = [item for item in candidate_results if item["family"] == family]
        family_summary[family] = {
            "candidate_count": len(members),
            "median_validation_timing_after_cost": float(
                np.median(
                    [
                        item["attribution"]["CONTINUOUS"]["validation"][
                            "timing_contribution_after_cost"
                        ]
                        for item in members
                    ]
                )
            ),
            "median_pseudo_oos_timing_after_cost": float(
                np.median(
                    [
                        item["attribution"]["CONTINUOUS"]["historical_pseudo_oos"][
                            "timing_contribution_after_cost"
                        ]
                        for item in members
                    ]
                )
            ),
            "median_placebo_percentile": float(
                np.median([item["placebo"]["placebo_percentile"] for item in members])
            ),
            "postmortem_gate_pass_count": int(sum(item["postmortem_gate_pass"] for item in members)),
        }

    passed = [item["id"] for item in candidate_results if item["postmortem_gate_pass"]]
    all_family_validation_timing_negative = all(
        summary["median_validation_timing_after_cost"] <= 0.0
        for summary in family_summary.values()
    )
    if not passed and all_family_validation_timing_negative:
        formal_conclusion = (
            "当前趋势、波动和估值状态族在验证期没有成本后动态择时能力。"
            "正式终止R6模型族，不启动R7，不研究相同变量的新阈值变体。"
        )
        failure_case = "CASE_A_NEGATIVE_DYNAMIC_TIMING"
    elif not passed:
        formal_conclusion = (
            "个别路径可能存在局部择时贡献，但没有候选同时通过匹配风险、安慰剂、稳定性和执行门槛。"
            "R6仍正式拒绝，不得激活前向。"
        )
        failure_case = "NO_CANDIDATE_PASSES_ALL_DIAGNOSTICS"
    else:
        formal_conclusion = (
            "历史归因存在继续观察依据，但历史不能批准模型；只能保留冻结的失败族观察档案，等待真正未来数据。"
        )
        failure_case = "DIAGNOSTIC_SIGNAL_ONLY_NO_APPROVAL"

    payload = {
        "protocol": protocol["protocol"],
        "periods": periods,
        "candidate_count": len(candidate_results),
        "candidates": candidate_results,
        "family_summary": family_summary,
        "split_inversion": split_inversion,
        "execution_mapping_rank_correlation": execution_rank_correlation,
        "boundaries": boundaries,
        "split_exposures": split_exposures,
        "decision": {
            "failure_case": failure_case,
            "postmortem_gate_pass_count": len(passed),
            "postmortem_gate_pass_candidates": passed,
            "formal_conclusion": formal_conclusion,
            "r6_forward_1_enabled": False,
            "live_position_mapping_enabled": False,
            "order_generation_enabled": False,
            "r7_allowed": False,
            "new_factor_research_enabled": False,
            "machine_learning_research_enabled": False,
        },
        "hashes": {
            PROTOCOL_FILE.name: _sha256(PROTOCOL_FILE),
            MATRIX_FILE.name: _sha256(MATRIX_FILE),
            MODEL_SELECTION_FILE.name: _sha256(MODEL_SELECTION_FILE),
            **{path.name: _sha256(path) for path in paths.values()},
        },
    }
    clean_payload = _clean(payload)
    OUTPUT_JSON.write_text(
        json.dumps(clean_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    ATTRIBUTION_REPORT.write_text(_render_attribution(clean_payload["candidates"]), encoding="utf-8")
    MATCHED_REPORT.write_text(_render_matched(clean_payload["candidates"]), encoding="utf-8")
    EVENT_REPORT.write_text(
        _render_events(
            events,
            model_selection["historical_validation_leader"],
            candidate_event_summaries,
        ),
        encoding="utf-8",
    )
    SPLIT_REPORT.write_text(
        _render_split(clean_payload["split_inversion"], boundaries, split_exposures),
        encoding="utf-8",
    )
    PLACEBO_REPORT.write_text(_render_placebo(clean_payload["candidates"]), encoding="utf-8")
    REJECTION_DECISION.write_text(_render_rejection(clean_payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": clean_payload["candidate_count"],
                "postmortem_gate_pass_count": clean_payload["decision"][
                    "postmortem_gate_pass_count"
                ],
                "failure_case": clean_payload["decision"]["failure_case"],
                "validation_pseudo_oos_spearman": clean_payload["split_inversion"][
                    "validation_pseudo_oos_spearman"
                ],
                "execution_mapping_rank_correlation": clean_payload[
                    "execution_mapping_rank_correlation"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
