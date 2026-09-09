"""结构信号功效与可识别性审计的统计计算。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少必需列：{missing}")


def build_paired_loss_panel(
    predictions: pd.DataFrame,
    *,
    target_specs: dict[str, dict[str, Any]],
    candidate_model: str,
    baseline_models: list[str],
    evaluation_start: str,
    evaluation_end: str,
) -> pd.DataFrame:
    """在共同 origin 上构造候选与基准的配对平方损失。"""

    _require_columns(
        predictions,
        ["origin", "target_id", "model_id", "prediction", "actual", "actual_status"],
        "冻结预测记录",
    )
    source = predictions.copy()
    source["origin"] = pd.to_datetime(source["origin"], errors="coerce").dt.normalize()
    source = source.loc[
        source["origin"].between(pd.Timestamp(evaluation_start), pd.Timestamp(evaluation_end))
        & source["target_id"].isin(target_specs)
        & source["model_id"].isin([candidate_model, *baseline_models])
        & source["prediction"].notna()
        & source["actual"].notna()
        & source["actual_status"].eq("ACTUAL_OBSERVED")
    ].copy()
    rows: list[pd.DataFrame] = []
    for target_id, target_spec in target_specs.items():
        target = source.loc[source["target_id"].eq(target_id)].copy()
        if target.empty:
            raise ValueError(f"{target_id}没有共同样本外预测")
        actual_check = target.groupby("origin")["actual"].agg(["min", "max"])
        if not np.allclose(
            actual_check["min"].to_numpy(dtype=float),
            actual_check["max"].to_numpy(dtype=float),
            equal_nan=False,
        ):
            raise ValueError(f"{target_id}在同一 origin 的 actual 不一致")
        actual = target.groupby("origin", as_index=True)["actual"].first()
        prediction_wide = target.pivot_table(
            index="origin", columns="model_id", values="prediction", aggfunc="first"
        )
        if candidate_model not in prediction_wide.columns:
            raise ValueError(f"{target_id}缺少候选模型预测")
        for baseline_model in baseline_models:
            if baseline_model not in prediction_wide.columns:
                raise ValueError(f"{target_id}缺少基准模型 {baseline_model}")
            paired = pd.DataFrame(
                {
                    "origin": prediction_wide.index,
                    "actual": actual.reindex(prediction_wide.index),
                    "candidate_prediction": prediction_wide[candidate_model],
                    "baseline_prediction": prediction_wide[baseline_model],
                }
            ).dropna()
            paired["candidate_squared_loss"] = (
                paired["candidate_prediction"] - paired["actual"]
            ) ** 2
            paired["baseline_squared_loss"] = (
                paired["baseline_prediction"] - paired["actual"]
            ) ** 2
            paired["loss_gain_vs_baseline"] = (
                paired["baseline_squared_loss"] - paired["candidate_squared_loss"]
            )
            paired["target_id"] = target_id
            paired["horizon_market_days"] = int(
                target_spec["horizon_market_days"]
            )
            paired["candidate_model"] = candidate_model
            paired["baseline_model"] = baseline_model
            rows.append(paired.reset_index(drop=True))
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(
        ["target_id", "baseline_model", "origin"], kind="stable"
    ).reset_index(drop=True)
    if result.empty:
        raise ValueError("配对损失面板为空")
    return result


def newey_west_long_run_variance(values: np.ndarray, max_lag: int) -> float:
    """使用 Bartlett 权重估计均值过程的长程方差。"""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    centered = array - float(array.mean())
    count = int(array.size)
    lag_limit = min(max(int(max_lag), 0), count - 1)
    long_run = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_limit + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1.0 - lag / (lag_limit + 1.0)
        long_run += 2.0 * weight * covariance
    return max(long_run, 0.0)


def build_nonoverlap_diagnostics(
    paired_loss: pd.DataFrame,
    *,
    target_specs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """按月度 origin 的相位抽取不重叠子样本并报告损失。"""

    rows: list[dict[str, Any]] = []
    for (target_id, baseline_model), group in paired_loss.groupby(
        ["target_id", "baseline_model"], sort=True
    ):
        ordered = group.sort_values("origin", kind="stable").reset_index(drop=True)
        block = int(target_specs[target_id]["monthly_overlap_block_origins"])
        for phase in range(block):
            sample = ordered.iloc[phase::block]
            baseline_mse = float(sample["baseline_squared_loss"].mean())
            candidate_mse = float(sample["candidate_squared_loss"].mean())
            relative_improvement = (
                1.0 - candidate_mse / baseline_mse
                if baseline_mse > 0.0
                else float("nan")
            )
            rows.append(
                {
                    "target_id": target_id,
                    "horizon_market_days": int(
                        target_specs[target_id]["horizon_market_days"]
                    ),
                    "baseline_model": baseline_model,
                    "block_origins": block,
                    "phase": phase,
                    "observation_count": int(len(sample)),
                    "first_origin": sample["origin"].min(),
                    "last_origin": sample["origin"].max(),
                    "baseline_mse": baseline_mse,
                    "candidate_mse": candidate_mse,
                    "relative_mse_improvement": relative_improvement,
                    "candidate_beats_baseline": bool(candidate_mse < baseline_mse),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target_id", "baseline_model", "phase"], kind="stable"
    ).reset_index(drop=True)


def _identifiability_status(
    total_years: float,
    *,
    short_horizon_years: float,
    acceptable_years: float,
) -> str:
    if not np.isfinite(total_years):
        return "NO_VIEW_POWER_NOT_ESTIMABLE"
    if total_years <= short_horizon_years:
        return "POTENTIALLY_IDENTIFIABLE_WITHIN_FIVE_YEARS"
    if total_years <= acceptable_years:
        return "LONG_HORIZON_FIVE_TO_TEN_YEARS"
    return "NOT_IDENTIFIABLE_WITHIN_TEN_YEARS"


def build_power_requirements(
    paired_loss: pd.DataFrame,
    *,
    target_specs: dict[str, dict[str, Any]],
    relative_effects: list[float],
    z_one_sided_alpha: float,
    z_target_power: float,
    origins_per_year: int,
    short_horizon_years: float,
    acceptable_years: float,
) -> pd.DataFrame:
    """估算不同相对 MSE 改善下所需的独立周期和保守月度样本。"""

    rows: list[dict[str, Any]] = []
    z_sum = float(z_one_sided_alpha) + float(z_target_power)
    for (target_id, baseline_model), group in paired_loss.groupby(
        ["target_id", "baseline_model"], sort=True
    ):
        ordered = group.sort_values("origin", kind="stable")
        gains = ordered["loss_gain_vs_baseline"].to_numpy(dtype=float)
        count = int(len(ordered))
        baseline_mse = float(ordered["baseline_squared_loss"].mean())
        candidate_mse = float(ordered["candidate_squared_loss"].mean())
        observed_relative_improvement = (
            1.0 - candidate_mse / baseline_mse
            if baseline_mse > 0.0
            else float("nan")
        )
        gain_standard_deviation = float(np.std(gains, ddof=1))
        gain_variance_population = float(np.var(gains, ddof=0))
        block = int(target_specs[target_id]["monthly_overlap_block_origins"])
        hac_lag = max(block - 1, 0)
        long_run_variance = newey_west_long_run_variance(gains, hac_lag)
        design_effect = (
            max(1.0, long_run_variance / gain_variance_population)
            if gain_variance_population > 0.0 and np.isfinite(long_run_variance)
            else float("nan")
        )
        current_effective_origins_hac = (
            count / design_effect if np.isfinite(design_effect) else float("nan")
        )
        current_nonoverlap_cycles = count / block
        for relative_effect in relative_effects:
            effect_absolute_mse = baseline_mse * float(relative_effect)
            estimable = (
                count >= 2
                and baseline_mse > 0.0
                and gain_standard_deviation > 0.0
                and effect_absolute_mse > 0.0
                and np.isfinite(design_effect)
            )
            if estimable:
                required_iid = max(
                    2,
                    int(
                        math.ceil(
                            (z_sum * gain_standard_deviation / effect_absolute_mse)
                            ** 2
                        )
                    ),
                )
                required_hac_origins = int(math.ceil(required_iid * design_effect))
                required_nonoverlap_origins = int(required_iid * block)
                conservative_required_origins = max(
                    required_hac_origins, required_nonoverlap_origins
                )
                total_years = conservative_required_origins / origins_per_year
                additional_years = max(
                    0.0, (conservative_required_origins - count) / origins_per_year
                )
                status = _identifiability_status(
                    total_years,
                    short_horizon_years=short_horizon_years,
                    acceptable_years=acceptable_years,
                )
            else:
                required_iid = None
                required_hac_origins = None
                required_nonoverlap_origins = None
                conservative_required_origins = None
                total_years = float("nan")
                additional_years = float("nan")
                status = "NO_VIEW_POWER_NOT_ESTIMABLE"
            rows.append(
                {
                    "target_id": target_id,
                    "horizon_market_days": int(
                        target_specs[target_id]["horizon_market_days"]
                    ),
                    "candidate_model": ordered["candidate_model"].iloc[0],
                    "baseline_model": baseline_model,
                    "observed_origin_count": count,
                    "first_origin": ordered["origin"].min(),
                    "last_origin": ordered["origin"].max(),
                    "baseline_mse": baseline_mse,
                    "candidate_mse": candidate_mse,
                    "observed_relative_mse_improvement": observed_relative_improvement,
                    "observed_candidate_beats_baseline": bool(candidate_mse < baseline_mse),
                    "hypothetical_relative_mse_improvement": float(relative_effect),
                    "effect_absolute_mse": effect_absolute_mse,
                    "loss_gain_standard_deviation": gain_standard_deviation,
                    "hac_lag_origins": hac_lag,
                    "hac_long_run_variance": long_run_variance,
                    "hac_design_effect": design_effect,
                    "current_effective_origins_hac": current_effective_origins_hac,
                    "overlap_block_origins": block,
                    "current_nonoverlap_cycles": current_nonoverlap_cycles,
                    "required_independent_cycles_iid": required_iid,
                    "required_monthly_origins_hac": required_hac_origins,
                    "required_monthly_origins_nonoverlap": required_nonoverlap_origins,
                    "conservative_required_monthly_origins": conservative_required_origins,
                    "conservative_total_calendar_years": total_years,
                    "conservative_additional_calendar_years": additional_years,
                    "identifiability_status": status,
                    "current_specification_requalification_allowed": False,
                    "portfolio_evaluation_allowed": False,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target_id", "baseline_model", "hypothetical_relative_mse_improvement"],
        kind="stable",
    ).reset_index(drop=True)


def realized_annualized_volatility(
    total_return_index: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    start: str,
    end: str,
    annualization_days: int,
) -> dict[str, Any]:
    """仅为夏普经济量级换算计算标的总回报指数波动率。"""

    _require_columns(total_return_index, [date_column, value_column], "总回报指数")
    frame = total_return_index[[date_column, value_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna().drop_duplicates(date_column, keep="last").sort_values(date_column)
    frame = frame.loc[
        frame[date_column].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    returns = frame[value_column].pct_change(fill_method=None).dropna()
    if len(returns) < annualization_days:
        raise ValueError("总回报指数样本不足一年，不能换算年化波动率")
    return {
        "start_date": frame[date_column].min().date().isoformat(),
        "end_date": frame[date_column].max().date().isoformat(),
        "price_observation_count": int(len(frame)),
        "return_observation_count": int(len(returns)),
        "annualization_days": int(annualization_days),
        "annualized_volatility": float(
            returns.std(ddof=1) * math.sqrt(annualization_days)
        ),
        "maximum_daily_return": float(returns.max()),
        "minimum_daily_return": float(returns.min()),
        "role": "ECONOMIC_SCALE_ONLY_NOT_STRATEGY_RETURN",
    }


def build_sharpe_identifiability(
    *,
    reference_sharpe: float,
    target_sharpe: float,
    realized_volatility: float,
    z_one_sided_alpha: float,
    z_target_power: float,
    dependence_multipliers: list[float],
) -> dict[str, Any]:
    """给出 Sharpe 跃迁的经济量级和高斯 IID 年数下界。"""

    delta = float(target_sharpe) - float(reference_sharpe)
    if delta <= 0.0 or realized_volatility <= 0.0:
        raise ValueError("Sharpe 目标或波动率参数无效")
    z_sum_squared = (float(z_one_sided_alpha) + float(z_target_power)) ** 2
    null_years = int(
        math.ceil(
            z_sum_squared
            * (1.0 + 0.5 * float(reference_sharpe) ** 2)
            / delta**2
        )
    )
    target_years = int(
        math.ceil(
            z_sum_squared
            * (1.0 + 0.5 * float(target_sharpe) ** 2)
            / delta**2
        )
    )
    scenarios = [
        {
            "dependence_variance_multiplier": float(multiplier),
            "required_calendar_years_lower_bound": int(
                math.ceil(target_years * float(multiplier))
            ),
        }
        for multiplier in dependence_multipliers
    ]
    return {
        "reference_sharpe": float(reference_sharpe),
        "target_sharpe": float(target_sharpe),
        "required_sharpe_increment": delta,
        "realized_annualized_volatility_scale": float(realized_volatility),
        "reference_gross_annual_excess_return_at_same_volatility": float(
            reference_sharpe * realized_volatility
        ),
        "target_gross_annual_excess_return_at_same_volatility": float(
            target_sharpe * realized_volatility
        ),
        "required_gross_annual_excess_return_increment_at_same_volatility": float(
            delta * realized_volatility
        ),
        "iid_gaussian_years_using_null_standard_error": null_years,
        "iid_gaussian_years_using_target_standard_error": target_years,
        "dependence_scenarios": scenarios,
        "status": "LOWER_BOUND_ONLY_NO_INVESTABLE_STRATEGY_RETURN_SERIES",
        "empirical_sharpe_test_allowed": False,
        "costs_included": False,
        "interpretation": (
            "该换算只说明经济量级和统计下界；当前没有通过验证的可投资收益序列，"
            "因此不能据此宣称 Sharpe 可实现或已被检验。"
        ),
    }
