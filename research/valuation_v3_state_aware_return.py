"""估值V3：处理重叠标签、状态时变、增长单次计价与经验回归速度。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr

from research.diagnose_valuation_negative_ic import add_forward_total_returns
from research.valuation_v2_expected_return import (
    BOND_FILE,
    INDEX_FILE,
    TOTAL_RETURN_FILE,
    VALUATION_FILE,
    _fair_pe_model,
    build_market_monthly,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "valuation_v3_state_aware_return.yaml"
V2_FILE = ROOT / "data" / "features" / "000300_valuation_v2_expected_return.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_valuation_v3_state_aware_return.parquet"
REPORT_JSON = ROOT / "reports" / "research" / "000300_valuation_v3_state_aware_return.json"
REPORT_MD = ROOT / "reports" / "research" / "000300_valuation_v3_state_aware_return.md"

HORIZONS = (20, 60, 120, 242)
QUINTILES = ("最低20%", "次低", "中性", "次高", "最高20%")
HISTORICAL_FAIR_PE_FEATURES = (
    "cgb_10y",
    "cgb_term_spread",
    "implied_roe",
    "eps_growth_12m",
    "realized_volatility_3m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_non_overlapping_queues(
    sample: pd.DataFrame,
    trading_dates: pd.Series,
    horizon_days: int,
    signal_column: str,
    target_column: str,
) -> list[dict[str, Any]]:
    """构造真正相隔至少h个交易日的错位队列，不把相邻标签当独立样本。"""

    dates = pd.Series(pd.to_datetime(trading_dates)).drop_duplicates().sort_values().reset_index(drop=True)
    position = {date: index for index, date in enumerate(dates)}
    data = sample[["date", signal_column, target_column]].dropna().copy()
    data["date"] = pd.to_datetime(data["date"])
    data["trading_position"] = data["date"].map(position)
    data = data.dropna(subset=["trading_position"]).sort_values("date").reset_index(drop=True)
    if data.empty:
        return []
    approximate_months = max(1, math.ceil(horizon_days / 21))
    rows: list[dict[str, Any]] = []
    for start_offset in range(min(approximate_months, len(data))):
        chosen: list[int] = []
        last_position: int | None = None
        for index in range(start_offset, len(data)):
            current_position = int(data.at[index, "trading_position"])
            if last_position is None or current_position - last_position >= horizon_days:
                chosen.append(index)
                last_position = current_position
        queue = data.loc[chosen]
        if len(queue) < 3:
            continue
        ic = (
            float(spearmanr(queue[signal_column], queue[target_column]).statistic)
            if len(queue) >= 4
            and queue[signal_column].nunique() > 1
            and queue[target_column].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "start_offset": start_offset,
                "observations": int(len(queue)),
                "first_date": str(queue["date"].min().date()),
                "last_date": str(queue["date"].max().date()),
                "spearman_ic": ic,
                "trading_positions": [int(value) for value in queue["trading_position"]],
            }
        )
    return rows


def newey_west_slope(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    max_lag: int,
) -> dict[str, float | int | None]:
    """带截距OLS斜率和Bartlett核Newey-West标准误。"""

    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    n = len(frame)
    if n < 4:
        return {"observations": n, "max_lag": max_lag, "slope": None}
    design = np.column_stack([np.ones(n), frame["x"].to_numpy()])
    target = frame["y"].to_numpy()
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ target
    residual = target - design @ beta
    meat = np.zeros((2, 2), dtype=float)
    for index in range(n):
        vector = design[index] * residual[index]
        meat += np.outer(vector, vector)
    effective_lag = min(max_lag, n - 1)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)
        gamma = np.zeros((2, 2), dtype=float)
        for index in range(lag, n):
            current = design[index] * residual[index]
            previous = design[index - lag] * residual[index - lag]
            gamma += np.outer(current, previous)
        meat += weight * (gamma + gamma.T)
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    t_stat = float(beta[1] / standard_error) if standard_error > 0 else np.nan
    p_value = float(2.0 * norm.sf(abs(t_stat))) if pd.notna(t_stat) else np.nan
    return {
        "observations": int(n),
        "max_lag": int(max_lag),
        "slope": float(beta[1]),
        "hac_standard_error": standard_error,
        "t_statistic": t_stat,
        "two_sided_p_value_normal_approximation": p_value,
    }


def moving_block_bootstrap_ic(
    signal: pd.Series,
    target: pd.Series,
    block_length: int,
    repetitions: int = 5000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """以连续月块重采样Spearman IC，保留重叠标签的序列相关。"""

    sample = pd.DataFrame({"signal": signal, "target": target}).dropna().reset_index(drop=True)
    n = len(sample)
    if n < 4:
        return {"observations": n, "repetitions": 0}
    length = min(max(1, block_length), n)
    starts = np.arange(0, n - length + 1)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    blocks_needed = math.ceil(n / length)
    for repetition in range(repetitions):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate([np.arange(start, start + length) for start in selected])[:n]
        draw = sample.iloc[indices]
        values[repetition] = spearmanr(draw["signal"], draw["target"]).statistic
    finite = values[np.isfinite(values)]
    return {
        "observations": int(n),
        "block_length_months": int(length),
        "repetitions": int(len(finite)),
        "median_ic": float(np.median(finite)),
        "probability_ic_positive": float(np.mean(finite > 0)),
        "ic_95pct_interval": [float(value) for value in np.quantile(finite, [0.025, 0.975])],
    }


def build_historical_fair_pe(
    market_monthly: pd.DataFrame,
    minimum_training_samples: int = 48,
) -> pd.DataFrame:
    """逐月只用更早历史产生同时点条件PE，用于估计回归速度。"""

    data = market_monthly.sort_values("date").reset_index(drop=True).copy()
    data["target_log_pe"] = np.log(data["pe_ttm"])
    predictions = np.full(len(data), np.nan)
    training_counts = np.zeros(len(data), dtype=int)
    for index in range(len(data)):
        train = data.iloc[:index].dropna(subset=[*HISTORICAL_FAIR_PE_FEATURES, "target_log_pe"])
        score = data.loc[[index], list(HISTORICAL_FAIR_PE_FEATURES)]
        if len(train) < minimum_training_samples or score.isna().any(axis=None):
            continue
        model = _fair_pe_model().fit(
            train[list(HISTORICAL_FAIR_PE_FEATURES)], train["target_log_pe"]
        )
        raw = float(model.predict(score)[0])
        lower, upper = train["target_log_pe"].quantile([0.10, 0.90])
        predictions[index] = float(np.exp(np.clip(raw, lower, upper)))
        training_counts[index] = len(train)
    return pd.DataFrame(
        {
            "date": data["date"],
            "pe_ttm": data["pe_ttm"],
            "historical_fair_pe": predictions,
            "fair_pe_training_samples": training_counts,
        }
    )


def estimate_walk_forward_reversion_speed(
    historical: pd.DataFrame,
    minimum_pairs: int = 12,
) -> pd.DataFrame:
    """每个信号日只用已完成的月度PE转移估计均值回归速度。"""

    data = historical.sort_values("date").reset_index(drop=True).copy()
    data["log_pe"] = np.log(data["pe_ttm"])
    data["log_fair_pe"] = np.log(data["historical_fair_pe"])
    data["fair_gap"] = data["log_fair_pe"] - data["log_pe"]
    data["next_log_pe_change"] = data["log_pe"].shift(-1) - data["log_pe"]
    data["transition_end_date"] = data["date"].shift(-1)
    rows: list[dict[str, Any]] = []
    for signal_date in data["date"]:
        train = data.loc[
            data["transition_end_date"].le(signal_date)
        ].dropna(subset=["fair_gap", "next_log_pe_change"])
        monthly_speed = np.nan
        if len(train) >= minimum_pairs and float(np.square(train["fair_gap"]).sum()) > 0:
            raw = float(
                np.dot(train["fair_gap"], train["next_log_pe_change"])
                / np.dot(train["fair_gap"], train["fair_gap"])
            )
            monthly_speed = float(np.clip(raw, 0.0, 1.0))
        annual_speed = (
            float(1.0 - (1.0 - monthly_speed) ** 12)
            if pd.notna(monthly_speed)
            else np.nan
        )
        rows.append(
            {
                "date": pd.Timestamp(signal_date),
                "monthly_reversion_speed": monthly_speed,
                "annual_reversion_speed": annual_speed,
                "reversion_training_pairs": int(len(train)),
                "reversion_training_last_end_date": (
                    pd.Timestamp(train["transition_end_date"].max()) if not train.empty else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows)


def estimate_component_annual_reversion_speed(
    component_panel: pd.DataFrame,
    minimum_pairs: int = 12,
) -> pd.DataFrame:
    """在同一成分正常化PE口径上估计未来12个月向FairPE的回归比例。"""

    data = component_panel[
        ["date", "component_normalized_pe", "conditioned_fair_pe_v2"]
    ].sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["fair_gap"] = np.log(
        data["conditioned_fair_pe_v2"] / data["component_normalized_pe"]
    )
    data["future_12m_pe_change"] = np.log(
        data["component_normalized_pe"].shift(-12) / data["component_normalized_pe"]
    )
    data["transition_end_date"] = data["date"].shift(-12)
    rows: list[dict[str, Any]] = []
    for signal_date in data["date"]:
        train = data.loc[data["transition_end_date"].le(signal_date)].dropna(
            subset=["fair_gap", "future_12m_pe_change"]
        )
        annual_speed = np.nan
        if len(train) >= minimum_pairs and float(np.square(train["fair_gap"]).sum()) > 0:
            raw = float(
                np.dot(train["fair_gap"], train["future_12m_pe_change"])
                / np.dot(train["fair_gap"], train["fair_gap"])
            )
            annual_speed = float(np.clip(raw, 0.0, 1.0))
        rows.append(
            {
                "date": pd.Timestamp(signal_date),
                "annual_reversion_speed": annual_speed,
                "reversion_training_pairs": int(len(train)),
                "reversion_training_last_end_date": (
                    pd.Timestamp(train["transition_end_date"].max()) if not train.empty else pd.NaT
                ),
                "reversion_estimation_basis": "COMPONENT_NORMALIZED_PE_12M_SAME_BASIS",
            }
        )
    return pd.DataFrame(rows)


def v3_expected_return(
    normalized_eps: float,
    current_index: float,
    current_normalized_pe: float,
    annual_growth: float,
    fair_pe: float,
    annual_dividend_yield: float,
    annual_cash_rate: float,
    horizon_days: int,
    annual_reversion_speed: float,
) -> dict[str, float]:
    """增长只进入未来EPS一次，PE只按历史估计速度部分回归。"""

    fraction = horizon_days / 242.0
    future_eps = normalized_eps * (1.0 + annual_growth) ** fraction
    horizon_speed = 1.0 - (1.0 - annual_reversion_speed) ** fraction
    target_pe = current_normalized_pe + horizon_speed * (fair_pe - current_normalized_pe)
    expected_future_index = future_eps * target_pe
    price_return = expected_future_index / current_index - 1.0
    dividend_return = (1.0 + annual_dividend_yield) ** fraction - 1.0
    total_return = price_return + dividend_return
    cash_return = (1.0 + annual_cash_rate) ** fraction - 1.0
    return {
        "future_normalized_eps": float(future_eps),
        "horizon_reversion_speed": float(horizon_speed),
        "target_pe": float(target_pe),
        "expected_future_index": float(expected_future_index),
        "expected_price_return": float(price_return),
        "expected_dividend_return": float(dividend_return),
        "expected_total_return": float(total_return),
        "cash_return": float(cash_return),
        "expected_excess_total_return": float(total_return - cash_return),
    }


def attach_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    """只用截至信号日状态构造固定市场分层。"""

    data = panel.sort_values("date").reset_index(drop=True).copy()
    data["rate_change_6m"] = data["cgb_10y"].diff(6)
    data["rate_regime"] = np.where(data["rate_change_6m"].ge(0), "利率上升", "利率下降")
    data.loc[data["rate_change_6m"].isna(), "rate_regime"] = pd.NA
    data["normalized_eps_growth_12m"] = data["component_normalized_index_eps"].pct_change(
        12, fill_method=None
    )
    data["earnings_regime"] = np.where(
        data["normalized_eps_growth_12m"].ge(0), "盈利上行", "盈利下行"
    )
    data.loc[data["normalized_eps_growth_12m"].isna(), "earnings_regime"] = pd.NA
    data["index_ma_10m"] = data["index_close"].rolling(10, min_periods=10).mean()
    data["index_momentum_3m"] = data["index_close"].pct_change(3, fill_method=None)
    bull = data["index_close"].gt(data["index_ma_10m"]) & data["index_momentum_3m"].gt(0)
    bear = data["index_close"].lt(data["index_ma_10m"]) & data["index_momentum_3m"].lt(0)
    data["market_regime"] = np.select([bull, bear], ["牛市", "熊市"], default="震荡")
    data.loc[data["index_ma_10m"].isna() | data["index_momentum_3m"].isna(), "market_regime"] = pd.NA
    data["risk_premium_proxy"] = (
        1.0 / data["component_normalized_pe"] - data["cgb_10y"] / 100.0
    )
    state_specs = {
        "risk_premium_regime": ("risk_premium_proxy", "高风险溢价", "低风险溢价"),
        "roe_regime": ("weighted_normalized_roe", "高ROE", "低ROE"),
        "volatility_regime": ("realized_volatility_3m", "高波动", "低波动"),
    }
    for output, (column, high_label, low_label) in state_specs.items():
        threshold = data[column].expanding(min_periods=12).median().shift(1)
        data[output] = np.where(data[column].ge(threshold), high_label, low_label)
        data.loc[threshold.isna() | data[column].isna(), output] = pd.NA
    return data


def build_v3_panel(
    v2_panel: pd.DataFrame,
    reversion_speed: pd.DataFrame,
    total_return_daily: pd.DataFrame,
) -> pd.DataFrame:
    """在V2点时基本面上应用V3收益恒等式并追加评价标签。"""

    panel = v2_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    stale_targets = [
        column
        for column in panel
        if column.startswith("forward_total_return_")
        or column.startswith("realized_excess_total_return_")
    ]
    panel = panel.drop(columns=stale_targets)
    speed = reversion_speed.copy()
    speed["date"] = pd.to_datetime(speed["date"]).astype("datetime64[ns]")
    panel = pd.merge_asof(
        panel.sort_values("date"),
        speed.sort_values("date"),
        on="date",
        direction="backward",
    )
    for horizon in HORIZONS:
        components = panel.apply(
            lambda row: v3_expected_return(
                normalized_eps=float(row["component_normalized_index_eps"]),
                current_index=float(row["index_close"]),
                current_normalized_pe=float(row["component_normalized_pe"]),
                annual_growth=float(row["expected_earnings_growth_annual"]),
                fair_pe=float(row["conditioned_fair_pe_v2"]),
                annual_dividend_yield=float(row["trailing_dividend_yield_annual"]),
                annual_cash_rate=float(row["annual_cash_rate"]),
                horizon_days=horizon,
                annual_reversion_speed=float(row["annual_reversion_speed"]),
            )
            if pd.notna(row["annual_reversion_speed"])
            else pd.Series(
                {
                    key: np.nan
                    for key in (
                        "future_normalized_eps",
                        "horizon_reversion_speed",
                        "target_pe",
                        "expected_future_index",
                        "expected_price_return",
                        "expected_dividend_return",
                        "expected_total_return",
                        "cash_return",
                        "expected_excess_total_return",
                    )
                }
            ),
            axis=1,
            result_type="expand",
        )
        for column in components:
            panel[f"v3_{column}_{horizon}d"] = components[column]
    panel = add_forward_total_returns(panel, total_return_daily, HORIZONS)
    for horizon in HORIZONS:
        panel[f"realized_excess_total_return_{horizon}d"] = (
            panel[f"forward_total_return_{horizon}d"] - panel[f"v3_cash_return_{horizon}d"]
        )
    return attach_regimes(panel)


def _quintiles(signal: pd.Series) -> pd.Series:
    output = pd.Series(pd.NA, index=signal.index, dtype="object")
    valid = signal.dropna()
    if len(valid) < 5:
        return output
    output.loc[valid.index] = pd.qcut(
        valid.rank(method="first"), 5, labels=QUINTILES
    ).astype(str)
    return output


def regime_ic_table(
    panel: pd.DataFrame,
    signal_column: str,
    target_column: str,
) -> list[dict[str, Any]]:
    """按固定点时状态报告IC；小样本直接暴露而不合并美化。"""

    regime_columns = (
        "rate_regime",
        "earnings_regime",
        "market_regime",
        "risk_premium_regime",
        "roe_regime",
        "volatility_regime",
    )
    rows: list[dict[str, Any]] = []
    for column in regime_columns:
        for state, group in panel[[column, signal_column, target_column]].dropna().groupby(column):
            ic = (
                float(spearmanr(group[signal_column], group[target_column]).statistic)
                if len(group) >= 4
                and group[signal_column].nunique() > 1
                and group[target_column].nunique() > 1
                else None
            )
            rows.append(
                {
                    "regime_dimension": column,
                    "state": str(state),
                    "observations": int(len(group)),
                    "spearman_ic": ic,
                    "mean_realized_edge": float(group[target_column].mean()),
                    "mean_signal": float(group[signal_column].mean()),
                }
            )
    return rows


def rolling_ic_summary(
    panel: pd.DataFrame,
    signal_column: str,
    target_column: str,
    window_months: int,
) -> dict[str, Any]:
    sample = panel[["date", signal_column, target_column]].dropna().sort_values("date")
    values: list[dict[str, Any]] = []
    for end in range(window_months, len(sample) + 1):
        block = sample.iloc[end - window_months : end]
        values.append(
            {
                "end_date": str(block["date"].iloc[-1].date()),
                "spearman_ic": float(
                    spearmanr(block[signal_column], block[target_column]).statistic
                ),
            }
        )
    if not values:
        return {"window_months": window_months, "window_count": 0, "series": []}
    array = np.asarray([item["spearman_ic"] for item in values], dtype=float)
    return {
        "window_months": window_months,
        "window_count": len(values),
        "minimum_ic": float(np.min(array)),
        "median_ic": float(np.median(array)),
        "latest_ic": float(array[-1]),
        "positive_window_ratio": float(np.mean(array > 0)),
        "series": values,
    }


def evaluate_v3_horizon(
    panel: pd.DataFrame,
    trading_dates: pd.Series,
    horizon: int,
) -> dict[str, Any]:
    signal = f"v3_expected_excess_total_return_{horizon}d"
    target = f"realized_excess_total_return_{horizon}d"
    sample = panel[["date", signal, target]].dropna().sort_values("date").copy()
    sample["quintile"] = _quintiles(sample[signal])
    grouped = (
        sample.groupby("quintile", observed=False)[target]
        .agg(["count", "mean", "median"])
        .reindex(QUINTILES)
    )
    queues = exact_non_overlapping_queues(
        sample, trading_dates, horizon, signal, target
    )
    queue_ics = [row["spearman_ic"] for row in queues if pd.notna(row["spearman_ic"])]
    lag = max(0, math.ceil(horizon / 21) - 1)
    split = len(sample) // 2
    halves = []
    for label, subset in (("前半段", sample.iloc[:split]), ("后半段", sample.iloc[split:])):
        halves.append(
            {
                "period": label,
                "observations": int(len(subset)),
                "start": str(subset["date"].min().date()),
                "end": str(subset["date"].max().date()),
                "spearman_ic": float(spearmanr(subset[signal], subset[target]).statistic),
            }
        )
    return {
        "horizon_days": horizon,
        "overlapping_monthly_observations": int(len(sample)),
        "spearman_ic": float(spearmanr(sample[signal], sample[target]).statistic),
        "newey_west": newey_west_slope(sample[signal], sample[target], lag),
        "moving_block_bootstrap": moving_block_bootstrap_ic(
            sample[signal], sample[target], max(1, math.ceil(horizon / 21))
        ),
        "exact_non_overlapping": {
            "warning": "不同起点队列是敏感性分析，彼此仍共享原始时间序列，不得把队列数当独立样本数。",
            "queue_count": len(queues),
            "observations_per_queue": [row["observations"] for row in queues],
            "queue_ic_median": float(np.median(queue_ics)) if queue_ics else None,
            "queue_ic_positive_ratio": float(np.mean(np.asarray(queue_ics) > 0)) if queue_ics else None,
            "queues": queues,
        },
        "chronological_halves": halves,
        "strictly_monotonic_quintile_means": bool(np.all(np.diff(grouped["mean"]) > 0)),
        "highest_minus_lowest_mean_realized_edge": float(
            grouped.loc["最高20%", "mean"] - grouped.loc["最低20%", "mean"]
        ),
        "quintiles": [
            {
                "state": label,
                "observations": int(grouped.loc[label, "count"]),
                "mean_realized_edge": float(grouped.loc[label, "mean"]),
                "median_realized_edge": float(grouped.loc[label, "median"]),
            }
            for label in QUINTILES
        ],
        "rolling_ic_24m": rolling_ic_summary(panel, signal, target, 24),
        "regime_ic": regime_ic_table(panel, signal, target),
    }


def realized_reversion_by_regime(
    panel: pd.DataFrame,
    horizon_months: int = 12,
) -> list[dict[str, Any]]:
    """事后度量不同状态下PE向FairPE收敛的实现比例，只用于机制归因。"""

    data = panel.sort_values("date").copy()
    data["future_normalized_pe"] = data["component_normalized_pe"].shift(-horizon_months)
    data["fair_gap_log"] = np.log(
        data["conditioned_fair_pe_v2"] / data["component_normalized_pe"]
    )
    data["realized_pe_change_log"] = np.log(
        data["future_normalized_pe"] / data["component_normalized_pe"]
    )
    data["realized_reversion_ratio"] = (
        data["realized_pe_change_log"] / data["fair_gap_log"]
    )
    material = data["fair_gap_log"].abs().ge(np.log(1.10))
    data = data.loc[material].replace([np.inf, -np.inf], np.nan)
    rows: list[dict[str, Any]] = []
    for dimension in (
        "rate_regime",
        "earnings_regime",
        "market_regime",
        "risk_premium_regime",
        "roe_regime",
        "volatility_regime",
    ):
        for state, group in data[[dimension, "realized_reversion_ratio"]].dropna().groupby(dimension):
            ratio = group["realized_reversion_ratio"]
            rows.append(
                {
                    "regime_dimension": dimension,
                    "state": str(state),
                    "observations": int(len(group)),
                    "median_realized_reversion_ratio": float(ratio.median()),
                    "positive_reversion_ratio": float(ratio.gt(0).mean()),
                    "full_or_more_reversion_ratio": float(ratio.ge(1).mean()),
                }
            )
    return rows


def same_sample_model_comparison(
    panel: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    target = f"realized_excess_total_return_{horizon}d"
    columns = {
        "simple_normalized_cheapness": "normalized_earnings_cheapness",
        "conditioned_normalized_cheapness": "conditioned_normalized_cheapness",
        "v2_expected_edge": f"expected_excess_total_return_{horizon}d",
        "v3_expected_edge": f"v3_expected_excess_total_return_{horizon}d",
    }
    available = [column for column in columns.values() if column in panel]
    sample = panel[[target, *available]].dropna()
    return {
        "horizon_days": horizon,
        "common_observations": int(len(sample)),
        "ics": {
            label: float(spearmanr(sample[column], sample[target]).statistic)
            for label, column in columns.items()
            if column in sample
        },
    }


def _pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 沪深300估值V3依赖性与状态稳定性审计",
        "",
        "> V3是看到V2结果后的诊断架构，不是预注册模型，不生成仓位。",
        "",
        "## 结论",
        "",
        f"- 状态：`{report['status']}`。",
        "- 242日相邻标签不得视为独立观测；报告同时给出真实非重叠队列、Newey-West和移动块Bootstrap。",
        f"- V3最新历史估计一年PE回归速度：{_pct(report['latest_signal']['annual_reversion_speed'])}。",
        f"- 被拒绝的跨口径vendor PE代理速度：{_pct(report['latest_signal']['rejected_vendor_proxy_annual_speed'])}。",
        f"- V3最新242日估值变化：{_pct(report['latest_signal']['valuation_change_242d'])}；V2为{_pct(report['latest_signal']['v2_valuation_change_242d'])}。",
        "- 外资流入/流出状态因没有点时外资流数据，本轮不做替代猜测。",
        "",
        "## 依赖性调整结果",
        "",
        "|期限|重叠月样本|普通IC|HAC斜率p值|块Bootstrap IC 95%区间|真实非重叠每队列样本|非重叠IC中位数|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["evaluations"]:
        interval = item["moving_block_bootstrap"]["ic_95pct_interval"]
        counts = item["exact_non_overlapping"]["observations_per_queue"]
        queue_median = item["exact_non_overlapping"]["queue_ic_median"]
        queue_median_text = "—" if queue_median is None else f"{queue_median:.3f}"
        lines.append(
            f"|{item['horizon_days']}日|{item['overlapping_monthly_observations']}|"
            f"{item['spearman_ic']:.3f}|{item['newey_west']['two_sided_p_value_normal_approximation']:.3f}|"
            f"[{interval[0]:.3f}, {interval[1]:.3f}]|"
            f"{min(counts) if counts else 0}–{max(counts) if counts else 0}|"
            f"{queue_median_text}|"
        )
    lines.extend(
        [
            "",
            "## 同样本模型对照",
            "",
            "|期限|简单正常化估值|条件正常化估值|V2 Edge|V3 Edge|",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["same_sample_comparison"]:
        ics = item["ics"]
        lines.append(
            f"|{item['horizon_days']}日|{ics['simple_normalized_cheapness']:.3f}|"
            f"{ics['conditioned_normalized_cheapness']:.3f}|{ics['v2_expected_edge']:.3f}|"
            f"{ics['v3_expected_edge']:.3f}|"
        )
    long_eval = next(item for item in report["evaluations"] if item["horizon_days"] == 242)
    lines.extend(
        [
            "",
        "## 242日时间稳定性",
            "",
            "|区间|样本|IC|",
            "|---|---:|---:|",
        ]
    )
    for row in long_eval["chronological_halves"]:
        lines.append(f"|{row['period']}|{row['observations']}|{row['spearman_ic']:.3f}|")
    rolling = long_eval["rolling_ic_24m"]
    lines.extend(
        [
            "",
            f"24个月rolling IC：最小{rolling.get('minimum_ic', float('nan')):.3f}，"
            f"中位数{rolling.get('median_ic', float('nan')):.3f}，"
            f"最新{rolling.get('latest_ic', float('nan')):.3f}，正值窗口占比{_pct(rolling.get('positive_window_ratio'))}。",
            "",
            "前后半段各自为正但全样本与rolling为负，属于明显的跨阶段水平错位；不能解释为稳定关系。",
            "",
            "## 242日状态IC",
            "",
            "|状态维度|状态|样本|IC|平均实际Edge|",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in long_eval["regime_ic"]:
        ic = "—" if row["spearman_ic"] is None else f"{row['spearman_ic']:.3f}"
        lines.append(
            f"|{row['regime_dimension']}|{row['state']}|{row['observations']}|"
            f"{ic}|{_pct(row['mean_realized_edge'])}|"
        )
    lines.extend(
        [
            "",
            "## 最新V3收益恒等式",
            "",
            f"- 信号日：{report['latest_signal']['date']}。",
            f"- 当前正常化PE：{report['latest_signal']['current_normalized_pe']:.2f}；FairPE：{report['latest_signal']['fair_pe']:.2f}。",
            f"- 一年经验回归速度：{_pct(report['latest_signal']['annual_reversion_speed'])}；目标PE：{report['latest_signal']['target_pe_242d']:.2f}。",
            f"- 未来正常化EPS：{report['latest_signal']['future_normalized_eps_242d']:.2f}；增长只在这里进入一次。",
            f"- 242日预期价格收益：{_pct(report['latest_signal']['expected_price_return_242d'])}；股息：{_pct(report['latest_signal']['expected_dividend_return_242d'])}；现金：{_pct(report['latest_signal']['cash_return_242d'])}；Edge：{_pct(report['latest_signal']['expected_edge_242d'])}。",
            "",
            "## 证据边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    index_daily = pd.read_parquet(INDEX_FILE)
    total_return = pd.read_parquet(TOTAL_RETURN_FILE)
    market_monthly = build_market_monthly(
        index_daily,
        total_return,
        pd.read_parquet(VALUATION_FILE),
        pd.read_parquet(BOND_FILE),
    )
    historical_fair = build_historical_fair_pe(market_monthly)
    proxy_speeds = estimate_walk_forward_reversion_speed(
        historical_fair,
        minimum_pairs=12,
    )
    v2_panel = pd.read_parquet(V2_FILE)
    speeds = estimate_component_annual_reversion_speed(
        v2_panel,
        minimum_pairs=int(config["mean_reversion"]["minimum_completed_annual_pairs"]),
    )
    panel = build_v3_panel(v2_panel, speeds, total_return)
    trading_dates = pd.to_datetime(total_return["date"]).drop_duplicates().sort_values()
    evaluations = [evaluate_v3_horizon(panel, trading_dates, horizon) for horizon in HORIZONS]
    comparisons = [same_sample_model_comparison(panel, horizon) for horizon in (60, 120, 242)]
    long_eval = next(item for item in evaluations if item["horizon_days"] == 242)
    bootstrap_interval = long_eval["moving_block_bootstrap"]["ic_95pct_interval"]
    time_stable = bool(
        long_eval["spearman_ic"] > 0
        and all(item["spearman_ic"] > 0 for item in long_eval["chronological_halves"])
        and long_eval["rolling_ic_24m"].get("latest_ic", -1.0) > 0
    )
    dependence_supported = bool(
        bootstrap_interval[0] > 0
        and long_eval["newey_west"]["two_sided_p_value_normal_approximation"] < 0.10
        and long_eval["exact_non_overlapping"]["queue_ic_positive_ratio"] is not None
        and long_eval["exact_non_overlapping"]["queue_ic_positive_ratio"] >= 0.60
        and min(long_eval["exact_non_overlapping"]["observations_per_queue"], default=0) >= 4
    )
    monotonic = all(
        item["strictly_monotonic_quintile_means"] for item in evaluations
    )
    latest = panel.dropna(subset=["annual_reversion_speed"]).sort_values("date").iloc[-1]
    latest_proxy_speed = (
        proxy_speeds.loc[
            proxy_speeds["date"].le(latest["date"])
            & proxy_speeds["annual_reversion_speed"].notna(),
            "annual_reversion_speed",
        ].iloc[-1]
    )
    latest_payload = {
        "date": str(pd.Timestamp(latest["date"]).date()),
        "current_normalized_pe": float(latest["component_normalized_pe"]),
        "fair_pe": float(latest["conditioned_fair_pe_v2"]),
        "annual_reversion_speed": float(latest["annual_reversion_speed"]),
        "rejected_vendor_proxy_annual_speed": float(latest_proxy_speed),
        "reversion_training_pairs": int(latest["reversion_training_pairs"]),
        "target_pe_242d": float(latest["v3_target_pe_242d"]),
        "future_normalized_eps_242d": float(latest["v3_future_normalized_eps_242d"]),
        "valuation_change_242d": float(
            latest["v3_target_pe_242d"] / latest["component_normalized_pe"] - 1.0
        ),
        "v2_valuation_change_242d": float(latest["expected_rerating_return_242d"]),
        "expected_price_return_242d": float(latest["v3_expected_price_return_242d"]),
        "expected_dividend_return_242d": float(latest["v3_expected_dividend_return_242d"]),
        "cash_return_242d": float(latest["v3_cash_return_242d"]),
        "expected_edge_242d": float(latest["v3_expected_excess_total_return_242d"]),
        "regimes": {
            column: (str(latest[column]) if pd.notna(latest[column]) else None)
            for column in (
                "rate_regime",
                "earnings_regime",
                "market_regime",
                "risk_premium_regime",
                "roe_regime",
                "volatility_regime",
            )
        },
    }
    report = {
        "status": (
            "POST_HOC_DIAGNOSTIC_PASSED_INTERNAL_GATES_FORWARD_TEST_REQUIRED"
            if time_stable and dependence_supported and monotonic
            else "POST_HOC_DIAGNOSTIC_ONLY_TIME_OR_DEPENDENCE_GATES_FAILED"
        ),
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "诊断V2重叠样本、时间失效、增长计价与PE回归速度；不做仓位。",
        "configuration": config,
        "sample": {
            "start": str(panel["date"].min().date()),
            "end": str(panel["date"].max().date()),
            "monthly_snapshots": int(len(panel)),
            "first_reversion_speed_date": str(
                panel.loc[panel["annual_reversion_speed"].notna(), "date"].min().date()
            ),
        },
        "evaluations": evaluations,
        "same_sample_comparison": comparisons,
        "realized_reversion_by_regime_242d": realized_reversion_by_regime(panel),
        "latest_signal": latest_payload,
        "acceptance": {
            "stable_positive_242d_overall_halves_and_latest_rolling_ic": time_stable,
            "dependence_adjusted_242d_evidence_supported": dependence_supported,
            "strictly_monotonic_quintiles_all_horizons": monotonic,
            "architecture_pre_registered": False,
            "trading_use_authorized": False,
        },
        "hypothesis_results": {
            "overlapping_242d_labels_inflate_confidence": "SUPPORTED_DEPENDENCE_ADJUSTED_INTERVAL_CROSSES_ZERO_NONOVERLAP_N3",
            "mean_reversion_is_state_dependent": "SUPPORTED_DESCRIPTIVE_SIGN_REVERSALS_BUT_TOO_FEW_TO_FIT",
            "growth_should_enter_future_eps_once": "IMPLEMENTED_STRUCTURALLY_NOT_VALIDATED_AS_ALPHA",
            "cross_basis_reversion_speed_is_invalid": "SUPPORTED_VENDOR_PROXY_78PCT_REJECTED_SAME_BASIS_34PCT",
            "fixed_state_independent_reversion_is_ready": "NOT_SUPPORTED",
            "tradable_510300_excess_is_established": "NOT_TESTED_AND_NOT_AUTHORIZED",
        },
        "researcher_degrees_of_freedom": {
            "architecture_selected_after_v2_results": True,
            "horizons_requested_before_v3_run": list(HORIZONS),
            "weights_searched": False,
            "state_thresholds_searched": False,
            "mean_reversion_bounds_searched": False,
            "interpretation": "V3只能用于形成下一轮冻结假设，不能把本轮结果升级为样本外证据。",
        },
        "limitations": [
            "V3因同口径回归速度需要先积累12个已完成年度转移，242日真实非重叠队列只剩3个观测，低于报告IC所需的最少4个。",
            "不同起点的非重叠队列来自同一条历史路径，不得把队列数当独立重复实验。",
            "HAC正态近似在约60个月小样本下仍不稳健，必须与移动块Bootstrap共同解释。",
            "状态拆分样本更小，当前只用于定位失效，不足以拟合状态专属参数。",
            "FairPE训练目标仍是历史市场PE，可能保留内生性；V3只消除了收益恒等式中的显式增长重复相加。",
            "被拒绝的vendor PE一月转移代理给出约78%年化速度；V3改用成分正常化PE同口径年度转移，但这些年度训练对仍高度重叠。",
            "没有点时外资流数据，因此外资流入/流出状态未检验。",
            "全部架构是在查看V2结果后形成，属于后验诊断而非预注册验证。",
            "IC与510300可交易超额仍有距离，本轮没有仓位、交易成本或执行回测。",
        ],
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "input_hashes": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                CONFIG_FILE,
                V2_FILE,
                INDEX_FILE,
                TOTAL_RETURN_FILE,
                VALUATION_FILE,
                BOND_FILE,
            )
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_FILE, index=False)
    report["output_sha256"] = _sha256(OUTPUT_FILE)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
