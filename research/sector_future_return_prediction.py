"""点时板块未来贡献目标、固定 Ridge 预测与历史否证。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.index_driver_attribution_v1_2 import (
    truncate_superseded_l1_intervals,
)
from research.index_driver_attribution_v1_3 import (
    restrict_industry_intervals_to_research_window,
)


@dataclass(frozen=True)
class PredictionRules:
    """板块未来贡献目标、模型与否证门槛。"""

    data_cutoff: pd.Timestamp
    horizon_days: int
    minimum_endpoint_coverage: float
    cash_annual_rate: float
    trading_days_per_year: int
    alpha: float
    minimum_training_dates: int
    numeric_features: tuple[str, ...]
    interaction_features: tuple[str, ...]
    minimum_calibration_errors: int
    calibration_quantiles: tuple[float, float, float]
    pseudo_oos_start: pd.Timestamp
    minimum_oos_predictions: int
    minimum_spearman: float
    bootstrap_repetitions: int
    bootstrap_block_length: int
    random_seed: int
    bootstrap_lower_bound: float
    minimum_mae_improvement_aggregate: float
    minimum_mae_improvement_mean: float
    minimum_direction_accuracy: float
    split_minimum_spearman: float
    etf_minimum_spearman: float
    minimum_calibrated_predictions: int
    minimum_brier_skill: float
    interval_coverage_minimum: float
    interval_coverage_maximum: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "PredictionRules":
        protocol = contract["protocol"]
        target = contract["target"]
        model = contract["model"]
        calibration = contract["calibration"]
        evaluation = contract["evaluation"]
        interval = evaluation["interval_80_coverage_range"]
        quantiles = calibration["error_quantiles"]
        return cls(
            data_cutoff=pd.Timestamp(protocol["historical_contamination_cutoff"]),
            horizon_days=int(target["horizon_trading_days"]),
            minimum_endpoint_coverage=float(
                target["minimum_endpoint_weight_coverage"]
            ),
            cash_annual_rate=float(target["cash_annual_rate"]),
            trading_days_per_year=int(target["trading_days_per_year"]),
            alpha=float(model["alpha"]),
            minimum_training_dates=int(
                model["minimum_mature_training_snapshot_dates"]
            ),
            numeric_features=tuple(model["numeric_features"]),
            interaction_features=tuple(model["industry_interaction_features"]),
            minimum_calibration_errors=int(
                calibration["minimum_prior_mature_oos_errors"]
            ),
            calibration_quantiles=(
                float(quantiles[0]),
                float(quantiles[1]),
                float(quantiles[2]),
            ),
            pseudo_oos_start=pd.Timestamp(evaluation["pseudo_oos_start"]),
            minimum_oos_predictions=int(
                evaluation["minimum_mature_oos_predictions"]
            ),
            minimum_spearman=float(evaluation["minimum_spearman"]),
            bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
            bootstrap_block_length=int(
                evaluation["bootstrap_block_length_months"]
            ),
            random_seed=int(evaluation["random_seed"]),
            bootstrap_lower_bound=float(
                evaluation["bootstrap_spearman_lower_bound_strictly_above"]
            ),
            minimum_mae_improvement_aggregate=float(
                evaluation["minimum_mae_improvement_vs_aggregate"]
            ),
            minimum_mae_improvement_mean=float(
                evaluation["minimum_mae_improvement_vs_expanding_mean"]
            ),
            minimum_direction_accuracy=float(
                evaluation["direction_accuracy_strictly_above"]
            ),
            split_minimum_spearman=float(
                evaluation["split_spearman_strictly_above"]
            ),
            etf_minimum_spearman=float(
                evaluation["etf_x60_spearman_strictly_above"]
            ),
            minimum_calibrated_predictions=int(
                evaluation["minimum_calibrated_predictions"]
            ),
            minimum_brier_skill=float(
                evaluation["brier_skill_strictly_above"]
            ),
            interval_coverage_minimum=float(interval[0]),
            interval_coverage_maximum=float(interval[1]),
        )


def _require_columns(
    frame: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name}缺少字段：{missing}")


def _prepare_intervals(
    intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    relevant = restrict_industry_intervals_to_research_window(
        intervals, start, end
    )
    resolved = truncate_superseded_l1_intervals(relevant)
    resolved["in_date"] = pd.to_datetime(resolved["in_date"])
    resolved["out_date"] = pd.to_datetime(resolved["out_date"])
    return resolved


def _active_industry(intervals: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    active = intervals.loc[
        intervals["in_date"].le(date)
        & (intervals["out_date"].isna() | intervals["out_date"].gt(date)),
        ["con_code", "industry_l1"],
    ].copy()
    if active["con_code"].duplicated().any():
        raise ValueError(f"{date.date()}存在重复有效行业")
    return active


def build_sector_forward_targets(
    sector_panel: pd.DataFrame,
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: PredictionRules,
) -> pd.DataFrame:
    """按信号日固定权重与行业，从下一交易日开盘构造60日贡献。"""

    _require_columns(
        sector_panel,
        {"date", "industry_l1", "sector_weight", "eligible_for_target_freeze"},
        "点时板块面板",
    )
    _require_columns(weights, {"trade_date", "con_code", "weight"}, "官方权重")
    _require_columns(
        constituent_daily,
        {"date", "con_code", "total_return_open", "total_return_close"},
        "成分股总收益日线",
    )
    panel = sector_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    point_weights = weights.copy()
    point_weights["trade_date"] = pd.to_datetime(point_weights["trade_date"])
    point_weights["snapshot_weight"] = pd.to_numeric(
        point_weights["weight"], errors="coerce"
    ) / 100.0
    daily = constituent_daily[
        ["date", "con_code", "total_return_open", "total_return_close"]
    ].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["total_return_open"] = pd.to_numeric(
        daily["total_return_open"], errors="coerce"
    )
    daily["total_return_close"] = pd.to_numeric(
        daily["total_return_close"], errors="coerce"
    )
    if daily[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股总收益日线存在重复证券日期")
    trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    daily_by_date = {
        pd.Timestamp(date): frame.set_index("con_code")
        for date, frame in daily.groupby("date", sort=False)
    }
    intervals = _prepare_intervals(
        industry_intervals, panel["date"].min(), rules.data_cutoff
    )
    rows: list[dict[str, Any]] = []
    for signal_date in sorted(panel["date"].unique()):
        signal_date = pd.Timestamp(signal_date)
        if signal_date not in trading_dates:
            continue
        signal_position = int(trading_dates.get_loc(signal_date))
        entry_position = signal_position + 1
        maturity_position = entry_position + rules.horizon_days - 1
        if maturity_position >= len(trading_dates):
            continue
        entry_date = pd.Timestamp(trading_dates[entry_position])
        maturity_date = pd.Timestamp(trading_dates[maturity_position])
        if maturity_date > rules.data_cutoff:
            continue
        snapshot = point_weights.loc[
            point_weights["trade_date"].eq(signal_date),
            ["con_code", "snapshot_weight"],
        ].copy()
        if snapshot.empty:
            raise ValueError(f"{signal_date.date()}缺少官方权重快照")
        active = _active_industry(intervals, signal_date)
        entry = daily_by_date[entry_date][["total_return_open"]].rename(
            columns={"total_return_open": "entry_total_return_open"}
        )
        maturity = daily_by_date[maturity_date][["total_return_close"]].rename(
            columns={"total_return_close": "maturity_total_return_close"}
        )
        cross = snapshot.merge(active, on="con_code", how="left", validate="one_to_one")
        cross = cross.merge(entry, left_on="con_code", right_index=True, how="left")
        cross = cross.merge(maturity, left_on="con_code", right_index=True, how="left")
        available = (
            cross["entry_total_return_open"].notna()
            & cross["entry_total_return_open"].gt(0)
            & cross["maturity_total_return_close"].notna()
            & cross["maturity_total_return_close"].gt(0)
            & cross["industry_l1"].notna()
        )
        endpoint_coverage = float(
            cross.loc[available, "snapshot_weight"].sum()
        )
        cross["component_return_60d"] = (
            cross["maturity_total_return_close"]
            / cross["entry_total_return_open"]
            - 1.0
        ).where(available)
        cross["component_contribution_60d"] = (
            cross["snapshot_weight"] * cross["component_return_60d"]
        )
        panel_date = panel.loc[panel["date"].eq(signal_date)].copy()
        contribution = (
            cross.loc[available]
            .groupby("industry_l1")["component_contribution_60d"]
            .sum(min_count=1)
        )
        snapshot_return = float(
            cross.loc[available, "component_contribution_60d"].sum()
        )
        status = (
            "TARGET_READY"
            if endpoint_coverage >= rules.minimum_endpoint_coverage
            else "NO_VIEW"
        )
        for sector in panel_date.itertuples(index=False):
            industry_name = str(sector.industry_l1)
            rows.append(
                {
                    "date": signal_date,
                    "entry_date": entry_date,
                    "maturity_date": maturity_date,
                    "industry_l1": industry_name,
                    "sector_weight": float(sector.sector_weight),
                    "sector_contribution_60d": float(
                        contribution.get(industry_name, 0.0)
                    ),
                    "snapshot_index_return_60d": snapshot_return,
                    "endpoint_weight_coverage": endpoint_coverage,
                    "target_output": status,
                    "failure_category": (
                        "PASS"
                        if status == "TARGET_READY"
                        else "ENDPOINT_WEIGHT_COVERAGE_BELOW_GATE"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "industry_l1"]).reset_index(
        drop=True
    )


def build_etf_forward_returns(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp],
    rules: PredictionRules,
) -> pd.DataFrame:
    """构造与板块目标同起点、同期限的510300含分红相对现金收益。"""

    _require_columns(etf_daily, {"date", "open", "low", "close"}, "510300日线")
    _require_columns(
        dividends, {"symbol", "ex_date", "cash_dividend_per_share"}, "510300分红"
    )
    market = etf_daily[["date", "open", "low", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    for column in ("open", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    if market.isna().any(axis=None) or (market[["open", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("510300日线存在空值或非正价格")
    market = market.loc[market["date"].le(rules.data_cutoff)].sort_values("date")
    market.reset_index(drop=True, inplace=True)
    positions = pd.Series(np.arange(len(market)), index=market["date"])
    cash = dividends.loc[dividends["symbol"].astype(str).eq("510300.SH")].copy()
    cash["ex_date"] = pd.to_datetime(cash["ex_date"])
    cash["cash_dividend_per_share"] = pd.to_numeric(
        cash["cash_dividend_per_share"], errors="coerce"
    )
    dividend_by_date = cash.groupby("ex_date")["cash_dividend_per_share"].sum()
    rows: list[dict[str, Any]] = []
    for value in signal_dates:
        signal_date = pd.Timestamp(value)
        signal_position = positions.get(signal_date)
        if signal_position is None:
            continue
        entry_position = int(signal_position) + 1
        maturity_position = entry_position + rules.horizon_days - 1
        if maturity_position >= len(market):
            continue
        entry = market.iloc[entry_position]
        wealth = float(entry["close"]) / float(entry["open"])
        first_cash = (1.0 + rules.cash_annual_rate) ** (
            1.0 / rules.trading_days_per_year
        )
        minimum_path = float(entry["low"]) / float(entry["open"]) - first_cash
        previous_close = float(entry["close"])
        for position in range(entry_position + 1, maturity_position + 1):
            day = market.iloc[position]
            dividend = float(dividend_by_date.get(pd.Timestamp(day["date"]), 0.0))
            low_wealth = wealth * (float(day["low"]) + dividend) / previous_close
            wealth = wealth * (float(day["close"]) + dividend) / previous_close
            elapsed = position - entry_position + 1
            cash_wealth = (1.0 + rules.cash_annual_rate) ** (
                elapsed / rules.trading_days_per_year
            )
            minimum_path = min(minimum_path, low_wealth - cash_wealth)
            previous_close = float(day["close"])
        cash_wealth = (1.0 + rules.cash_annual_rate) ** (
            rules.horizon_days / rules.trading_days_per_year
        )
        rows.append(
            {
                "date": signal_date,
                "etf_entry_date": pd.Timestamp(entry["date"]),
                "etf_maturity_date": pd.Timestamp(
                    market.iloc[maturity_position]["date"]
                ),
                "etf_x60": float(wealth - cash_wealth),
                "etf_min_path60": float(minimum_path),
            }
        )
    return pd.DataFrame(rows)


def _mature_training_dates(targets: pd.DataFrame, signal_date: pd.Timestamp) -> list[pd.Timestamp]:
    dates = targets.loc[
        targets["maturity_date"].lt(signal_date)
        & targets["target_output"].eq("TARGET_READY"),
        "date",
    ].drop_duplicates()
    return sorted(pd.Timestamp(value) for value in dates)


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """退化样本返回 NaN，避免把未定义相关性误当成统计证据。"""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    left_valid = left_array[valid]
    right_valid = right_array[valid]
    if (
        len(left_valid) < 2
        or np.unique(left_valid).size < 2
        or np.unique(right_valid).size < 2
    ):
        return np.nan
    return float(spearmanr(left_valid, right_valid).statistic)


def _feature_engineering(
    frame: pd.DataFrame,
    raw_features: Sequence[str],
    interaction_features: Sequence[str],
    industries: Sequence[str],
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    weight = frame["sector_weight"].astype(float)
    result["global::weight"] = weight
    for feature in raw_features:
        if feature == "sector_weight":
            continue
        result[f"global::{feature}"] = weight * frame[feature].astype(float)
    date_totals = frame.groupby("date")["sector_weight"].transform("sum")
    for feature in interaction_features:
        weighted = weight * frame[feature].astype(float)
        aggregate = weighted.groupby(frame["date"]).transform("sum") / date_totals
        relative = frame[feature].astype(float) - aggregate
        for industry in industries:
            mask = frame["industry_l1"].astype(str).eq(industry).astype(float)
            result[f"industry::{industry}::{feature}"] = weight * relative * mask
    return result


def _training_median_impute(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅用训练窗口中位数插补，并拒绝训练期全空特征。"""

    train_result = train.copy()
    score_result = score.copy()
    medians = train_result.loc[:, columns].median(axis=0, skipna=True)
    all_missing = sorted(str(column) for column in medians.index[medians.isna()])
    if all_missing:
        raise ValueError(f"训练窗口特征全缺失，不能插补：{all_missing}")
    train_result.loc[:, columns] = train_result.loc[:, columns].fillna(medians)
    score_result.loc[:, columns] = score_result.loc[:, columns].fillna(medians)
    if score_result.loc[:, columns].isna().any(axis=None):
        raise ValueError("评分窗口仍存在无法插补的特征")
    return train_result, score_result


def _sector_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    rules: PredictionRules,
) -> tuple[np.ndarray, dict[str, Any]]:
    industries = sorted(train["industry_l1"].astype(str).unique())
    raw_columns = sorted(set(rules.numeric_features) - {"sector_weight"})
    prepared_train, prepared_score = _training_median_impute(
        train, score, raw_columns
    )
    design_train = _feature_engineering(
        prepared_train,
        rules.numeric_features,
        rules.interaction_features,
        industries,
    )
    design_score = _feature_engineering(
        prepared_score,
        rules.numeric_features,
        rules.interaction_features,
        industries,
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=rules.alpha, fit_intercept=False)),
        ]
    )
    pipeline.fit(design_train, train["sector_contribution_60d"].astype(float))
    prediction = pipeline.predict(design_score)
    return prediction, {
        "training_rows": int(len(train)),
        "design_columns": int(design_train.shape[1]),
        "training_industries": int(len(industries)),
    }


def _aggregate_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        weight = group["sector_weight"].astype(float)
        total = float(weight.sum())
        row: dict[str, Any] = {"date": pd.Timestamp(date)}
        for feature in features:
            if feature == "sector_weight":
                continue
            valid = group[feature].notna() & weight.gt(0)
            valid_weight = float(weight.loc[valid].sum())
            row[feature] = (
                float(np.average(group.loc[valid, feature], weights=weight.loc[valid]))
                if valid_weight > 0
                else np.nan
            )
        row["snapshot_index_return_60d"] = float(
            group["snapshot_index_return_60d"].iloc[0]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_model_prediction(
    train: pd.DataFrame, score: pd.DataFrame, rules: PredictionRules
) -> float:
    feature_columns = [
        feature for feature in rules.numeric_features if feature != "sector_weight"
    ]
    train_aggregate = _aggregate_frame(train, rules.numeric_features)
    score_aggregate = _aggregate_frame(score, rules.numeric_features)
    prepared_train, prepared_score = _training_median_impute(
        train_aggregate, score_aggregate, feature_columns
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=rules.alpha)),
        ]
    )
    pipeline.fit(
        prepared_train[feature_columns],
        prepared_train["snapshot_index_return_60d"],
    )
    return float(pipeline.predict(prepared_score[feature_columns])[0])


def walk_forward_sector_forecasts(
    sector_panel: pd.DataFrame,
    targets: pd.DataFrame,
    etf_targets: pd.DataFrame,
    rules: PredictionRules,
) -> pd.DataFrame:
    """只用已成熟目标逐月拟合唯一板块模型和聚合基准。"""

    features = sector_panel.copy()
    features["date"] = pd.to_datetime(features["date"])
    target = targets.copy()
    target["date"] = pd.to_datetime(target["date"])
    target["maturity_date"] = pd.to_datetime(target["maturity_date"])
    merged = features.merge(
        target,
        on=["date", "industry_l1", "sector_weight"],
        how="left",
        validate="one_to_one",
    )
    eligible = merged.loc[
        merged["eligible_for_target_freeze"].astype(bool)
        & merged["target_output"].eq("TARGET_READY")
    ].copy()
    etf_lookup = etf_targets.set_index("date") if not etf_targets.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for signal_date in sorted(eligible["date"].unique()):
        signal_date = pd.Timestamp(signal_date)
        if signal_date < rules.pseudo_oos_start:
            continue
        mature_dates = _mature_training_dates(target, signal_date)
        if len(mature_dates) < rules.minimum_training_dates:
            continue
        train = eligible.loc[eligible["date"].isin(mature_dates)].copy()
        score = eligible.loc[eligible["date"].eq(signal_date)].copy()
        if score.empty:
            continue
        sector_prediction, audit = _sector_model(train, score, rules)
        predicted_index = float(np.sum(sector_prediction))
        aggregate_prediction = _aggregate_model_prediction(train, score, rules)
        actual_index = float(score["snapshot_index_return_60d"].iloc[0])
        maturity_date = pd.Timestamp(score["maturity_date"].iloc[0])
        training_index = (
            train[["date", "snapshot_index_return_60d"]]
            .drop_duplicates("date")
            .sort_values("date")
        )
        expanding_mean = float(training_index["snapshot_index_return_60d"].mean())
        expanding_positive_probability = float(
            training_index["snapshot_index_return_60d"].gt(0).mean()
        )
        prior_errors = [
            float(row["actual_index_return_60d"] - row["predicted_index_return_60d"])
            for row in rows
            if pd.Timestamp(row["maturity_date"]) < signal_date
            and pd.notna(row["actual_index_return_60d"])
        ]
        if len(prior_errors) >= rules.minimum_calibration_errors:
            errors = np.asarray(prior_errors, dtype=float)
            quantiles = np.quantile(errors, rules.calibration_quantiles)
            lower = predicted_index + float(quantiles[0])
            median = predicted_index + float(quantiles[1])
            upper = predicted_index + float(quantiles[2])
            positive_probability = float(np.mean(predicted_index + errors > 0))
            calibration_status = "CALIBRATED_PRIOR_OOS_ERRORS"
        else:
            lower = median = upper = positive_probability = np.nan
            calibration_status = "INSUFFICIENT_PRIOR_MATURE_OOS_ERRORS"
        etf_x60 = np.nan
        etf_min_path60 = np.nan
        if not etf_targets.empty and signal_date in etf_lookup.index:
            etf_x60 = float(etf_lookup.at[signal_date, "etf_x60"])
            etf_min_path60 = float(etf_lookup.at[signal_date, "etf_min_path60"])
        rows.append(
            {
                "date": signal_date,
                "maturity_date": maturity_date,
                "training_snapshot_dates": int(len(mature_dates)),
                **audit,
                "predicted_index_return_60d": predicted_index,
                "aggregate_baseline_prediction_60d": aggregate_prediction,
                "expanding_mean_prediction_60d": expanding_mean,
                "actual_index_return_60d": actual_index,
                "etf_x60": etf_x60,
                "etf_min_path60": etf_min_path60,
                "calibration_status": calibration_status,
                "prior_mature_oos_error_count": int(len(prior_errors)),
                "predicted_return_q10": lower,
                "predicted_return_q50": median,
                "predicted_return_q90": upper,
                "predicted_positive_probability": positive_probability,
                "baseline_positive_probability": expanding_positive_probability,
                "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
                "research_output": "PREDICTION_AUDIT_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _moving_block_bootstrap_spearman(
    prediction: np.ndarray,
    actual: np.ndarray,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    n = len(actual)
    if n < 4:
        return {"observations": n, "repetitions": 0, "interval_95pct": None}
    length = min(max(1, block_length), n)
    starts = np.arange(0, n - length + 1)
    blocks_needed = int(np.ceil(n / length))
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + length) for start in selected]
        )[:n]
        statistic = _safe_spearman(prediction[indices], actual[indices])
        values[repetition] = statistic
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "observations": n,
            "block_length_months": length,
            "repetitions": 0,
            "median_spearman": None,
            "interval_95pct": None,
        }
    interval = np.quantile(finite, [0.025, 0.975])
    return {
        "observations": n,
        "block_length_months": length,
        "repetitions": int(len(finite)),
        "median_spearman": float(np.median(finite)),
        "interval_95pct": [float(interval[0]), float(interval[1])],
    }


def evaluate_sector_forecasts(
    forecasts: pd.DataFrame, rules: PredictionRules
) -> dict[str, Any]:
    """按冻结门槛评价唯一S1模型，不生成仓位或订单。"""

    mature = forecasts.loc[
        forecasts["maturity_date"].le(rules.data_cutoff)
        & forecasts["actual_index_return_60d"].notna()
    ].sort_values("date")
    if mature.empty:
        raise ValueError("没有成熟的板块未来贡献预测")
    prediction = mature["predicted_index_return_60d"].to_numpy(float)
    actual = mature["actual_index_return_60d"].to_numpy(float)
    aggregate = mature["aggregate_baseline_prediction_60d"].to_numpy(float)
    mean_prediction = mature["expanding_mean_prediction_60d"].to_numpy(float)
    spearman = _safe_spearman(prediction, actual)
    etf_valid = mature[["predicted_index_return_60d", "etf_x60"]].dropna()
    etf_spearman = (
        _safe_spearman(
            etf_valid["predicted_index_return_60d"], etf_valid["etf_x60"]
        )
        if len(etf_valid) >= 4
        else np.nan
    )
    model_mae = float(np.mean(np.abs(prediction - actual)))
    aggregate_mae = float(np.mean(np.abs(aggregate - actual)))
    mean_mae = float(np.mean(np.abs(mean_prediction - actual)))
    improvement_aggregate = 1.0 - model_mae / aggregate_mae
    improvement_mean = 1.0 - model_mae / mean_mae
    direction_accuracy = float(np.mean((prediction > 0) == (actual > 0)))
    midpoint = len(mature) // 2
    first_spearman = _safe_spearman(prediction[:midpoint], actual[:midpoint])
    second_spearman = _safe_spearman(prediction[midpoint:], actual[midpoint:])
    bootstrap = _moving_block_bootstrap_spearman(
        prediction,
        actual,
        rules.bootstrap_block_length,
        rules.bootstrap_repetitions,
        rules.random_seed,
    )
    calibrated = mature.loc[
        mature["predicted_positive_probability"].notna()
        & mature["predicted_return_q10"].notna()
        & mature["predicted_return_q90"].notna()
    ].copy()
    if calibrated.empty:
        brier_skill = np.nan
        interval_coverage = np.nan
    else:
        outcome = calibrated["actual_index_return_60d"].gt(0).astype(float)
        model_brier = float(
            np.mean(np.square(calibrated["predicted_positive_probability"] - outcome))
        )
        baseline_brier = float(
            np.mean(np.square(calibrated["baseline_positive_probability"] - outcome))
        )
        brier_skill = 1.0 - model_brier / baseline_brier if baseline_brier > 0 else np.nan
        interval_coverage = float(
            (
                calibrated["actual_index_return_60d"]
                .ge(calibrated["predicted_return_q10"])
                & calibrated["actual_index_return_60d"].le(
                    calibrated["predicted_return_q90"]
                )
            ).mean()
        )
    bootstrap_lower = (
        bootstrap["interval_95pct"][0]
        if bootstrap["interval_95pct"] is not None
        else np.nan
    )
    gates = {
        "minimum_mature_oos_predictions": len(mature)
        >= rules.minimum_oos_predictions,
        "minimum_spearman": pd.notna(spearman)
        and spearman >= rules.minimum_spearman,
        "bootstrap_spearman_lower_bound": pd.notna(bootstrap_lower)
        and bootstrap_lower > rules.bootstrap_lower_bound,
        "mae_improvement_vs_aggregate": improvement_aggregate
        >= rules.minimum_mae_improvement_aggregate,
        "mae_improvement_vs_expanding_mean": improvement_mean
        >= rules.minimum_mae_improvement_mean,
        "direction_accuracy": direction_accuracy
        > rules.minimum_direction_accuracy,
        "first_half_spearman": pd.notna(first_spearman)
        and first_spearman > rules.split_minimum_spearman,
        "second_half_spearman": pd.notna(second_spearman)
        and second_spearman > rules.split_minimum_spearman,
        "etf_x60_spearman": pd.notna(etf_spearman)
        and etf_spearman > rules.etf_minimum_spearman,
        "minimum_calibrated_predictions": len(calibrated)
        >= rules.minimum_calibrated_predictions,
        "brier_skill": pd.notna(brier_skill)
        and brier_skill > rules.minimum_brier_skill,
        "interval_80_coverage": pd.notna(interval_coverage)
        and rules.interval_coverage_minimum
        <= interval_coverage
        <= rules.interval_coverage_maximum,
    }
    status = (
        "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        if all(gates.values())
        else "HISTORICAL_REJECTED_FROZEN"
    )
    return {
        "status": status,
        "candidate_id": "S1_SECTOR_FUNDAMENTAL_RIDGE_V1",
        "mature_oos_predictions": int(len(mature)),
        "calibrated_predictions": int(len(calibrated)),
        "first_prediction_date": str(mature["date"].min().date()),
        "last_mature_prediction_date": str(mature["date"].max().date()),
        "spearman": spearman,
        "bootstrap_spearman": bootstrap,
        "model_mae": model_mae,
        "aggregate_baseline_mae": aggregate_mae,
        "expanding_mean_mae": mean_mae,
        "mae_improvement_vs_aggregate": improvement_aggregate,
        "mae_improvement_vs_expanding_mean": improvement_mean,
        "direction_accuracy": direction_accuracy,
        "first_half_spearman": first_spearman,
        "second_half_spearman": second_spearman,
        "etf_x60_spearman": etf_spearman,
        "brier_skill": brier_skill,
        "interval_80_coverage": interval_coverage,
        "gates": gates,
        "safety": {
            "forecast_eligible_emitted": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }


__all__ = [
    "PredictionRules",
    "build_etf_forward_returns",
    "build_sector_forward_targets",
    "evaluate_sector_forecasts",
    "walk_forward_sector_forecasts",
]
