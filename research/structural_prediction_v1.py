"""510300 结构预测 V1 的点时特征、滚动预测与样本外评估。

本模块只生成预测研究结果，不生成仓位、净值、夏普率或交易指令。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score


PASS_STATE_STATUS = "PASS_FROZEN_PARENT_STATES"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少必需字段：{missing}")


def normalize_dates(values: pd.Series) -> pd.Series:
    result = pd.to_datetime(values, errors="raise")
    if result.dt.tz is not None:
        result = result.dt.tz_convert(None)
    return result.astype("datetime64[ns]").dt.normalize()


def _validate_unique_dates(frame: pd.DataFrame, column: str, name: str) -> None:
    duplicates = frame.loc[frame[column].duplicated(keep=False), column]
    if not duplicates.empty:
        raise ValueError(
            f"{name} 的 {column} 存在重复值："
            f"{duplicates.astype(str).head(5).tolist()}"
        )


def build_prediction_state_panel(
    condition_state: pd.DataFrame,
    present_value: pd.DataFrame,
    old_r5_valuation_signals: pd.DataFrame,
    *,
    old_r5_asof_tolerance_calendar_days: int,
) -> pd.DataFrame:
    """在不读取 H00300 全收益值的前提下冻结结构状态和旧估值基准。"""

    condition_required = [
        "origin",
        "condition_quality_status",
        "cf_level",
        "cf_news",
        "cf_breadth",
        "cf_concentration",
        "cash_flow_expectation_factor",
        "cross_sectional_expected_return_factor",
        "equity_risk_premium_expanding_z",
        "discount_rate_easing_news",
        "valuation_concentration",
        "duration_compression",
        "risk_bearing_capacity_factor",
        "risk_capacity_news",
        "member_positive_breadth_60d",
        "large_weight_minus_median_member_return_60d",
        "industry_negative_breadth_20d",
    ]
    pv_required = [
        "origin",
        "cap_weighted_positive_earnings_yield",
        "cap_weighted_book_to_price",
        "cap_weighted_sales_to_price",
    ]
    old_required = [
        "date",
        "index_close",
        "pe_ttm",
        "pb",
        "fair_mid_12m",
        "target_position",
        "valuation_state",
    ]
    _require_columns(condition_state, condition_required, "第二阶段条件状态")
    _require_columns(present_value, pv_required, "父项目现值状态")
    _require_columns(old_r5_valuation_signals, old_required, "旧 R5 估值信号")

    state = condition_state.copy()
    pv = present_value[pv_required].copy()
    old = old_r5_valuation_signals[old_required].copy()
    state["origin"] = normalize_dates(state["origin"])
    pv["origin"] = normalize_dates(pv["origin"])
    old["date"] = normalize_dates(old["date"])
    _validate_unique_dates(state, "origin", "第二阶段条件状态")
    _validate_unique_dates(pv, "origin", "父项目现值状态")
    _validate_unique_dates(old, "date", "旧 R5 估值信号")

    state = state.merge(pv, on="origin", how="left", validate="one_to_one")
    state = pd.merge_asof(
        state.sort_values("origin"),
        old.sort_values("date").rename(columns={"date": "old_r5_asof_date"}),
        left_on="origin",
        right_on="old_r5_asof_date",
        direction="backward",
        tolerance=pd.Timedelta(days=int(old_r5_asof_tolerance_calendar_days)),
    ).reset_index(drop=True)
    if (state["old_r5_asof_date"] > state["origin"]).fillna(False).any():
        raise ValueError("旧 R5 估值 as-of 合并读取了 origin 之后的数据")
    state["old_r5_asof_age_calendar_days"] = (
        state["origin"] - state["old_r5_asof_date"]
    ).dt.days
    numeric_old = ["index_close", "pe_ttm", "pb", "fair_mid_12m"]
    state[numeric_old] = state[numeric_old].apply(pd.to_numeric, errors="coerce")
    state["old_r5_valuation_gap_mid_12m"] = (
        state["fair_mid_12m"] / state["index_close"] - 1.0
    )
    state["old_r5_earnings_yield"] = 1.0 / state["pe_ttm"]
    state["old_r5_book_to_price"] = 1.0 / state["pb"]
    old_ready = state[
        [
            "old_r5_valuation_gap_mid_12m",
            "old_r5_earnings_yield",
            "old_r5_book_to_price",
        ]
    ].notna().all(axis=1)
    state["old_r5_baseline_status"] = np.where(
        old_ready,
        "PASS_FIXED_REJECTED_R5_BASELINE_FEATURES",
        "NO_VIEW_OLD_R5_BASELINE_FEATURES",
    )

    state["cf_news_x_dr_easing_news"] = (
        state["cf_news"] * state["discount_rate_easing_news"]
    )
    state["dr_easing_news_x_rc_news"] = (
        state["discount_rate_easing_news"] * state["risk_capacity_news"]
    )
    state["cf_level_x_erp_z"] = (
        state["cf_level"] * state["equity_risk_premium_expanding_z"]
    )
    state["erp_z_x_rc_factor"] = (
        state["equity_risk_premium_expanding_z"]
        * state["risk_bearing_capacity_factor"]
    )
    state["h00300_values_read"] = False
    state["portfolio_evaluation_allowed"] = False
    state["model_position_target"] = "UNSET"
    return state


def attach_price_features_and_targets(
    prediction_state: pd.DataFrame,
    total_return: pd.DataFrame,
    *,
    observation_cutoff: str | pd.Timestamp,
    return_horizons_market_days: Sequence[int],
    drawdown_horizon_market_days: int,
    drawdown_threshold: float,
    trading_days_per_year: int = 242,
    maximum_origin_gap_calendar_days: int = 5,
) -> pd.DataFrame:
    """使用每个 origin 当日及以前价格构造基准特征，并追加冻结目标。"""

    _require_columns(prediction_state, ["origin"], "预测状态面板")
    _require_columns(total_return, ["date", "close"], "H00300 全收益序列")
    result = prediction_state.copy()
    result["origin"] = normalize_dates(result["origin"])
    daily = total_return[["date", "close"]].copy()
    daily["date"] = normalize_dates(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.loc[
        daily["date"].le(pd.Timestamp(observation_cutoff).normalize())
    ].sort_values("date")
    _validate_unique_dates(daily, "date", "H00300 全收益序列")
    if daily.empty or daily["close"].isna().any() or daily["close"].le(0).any():
        raise ValueError("H00300 全收益 close 缺失、非正或为空")

    dates = pd.DatetimeIndex(daily["date"])
    closes = daily["close"].to_numpy(dtype=float)
    daily_returns = np.full(len(closes), np.nan, dtype=float)
    daily_returns[1:] = closes[1:] / closes[:-1] - 1.0
    positions: list[int] = []
    asof_dates: list[pd.Timestamp | pd.NaT] = []
    statuses: list[str] = []
    for origin in result["origin"]:
        position = int(dates.searchsorted(origin, side="right") - 1)
        if position < 0:
            positions.append(-1)
            asof_dates.append(pd.NaT)
            statuses.append("NO_VIEW_NO_H00300_ON_OR_BEFORE_ORIGIN")
            continue
        age_days = int((origin - dates[position]).days)
        if age_days > int(maximum_origin_gap_calendar_days):
            positions.append(-1)
            asof_dates.append(pd.NaT)
            statuses.append("NO_VIEW_H00300_ORIGIN_GAP_TOO_OLD")
            continue
        positions.append(position)
        asof_dates.append(dates[position])
        statuses.append("PASS_H00300_AT_ORIGIN")
    result["h00300_asof_date"] = asof_dates
    result["h00300_origin_status"] = statuses

    for horizon in (20, 60):
        trailing_values: list[float] = []
        volatility_values: list[float] = []
        feature_statuses: list[str] = []
        for position in positions:
            start = position - horizon
            if position < 0 or start < 0:
                trailing_values.append(np.nan)
                volatility_values.append(np.nan)
                feature_statuses.append("NO_VIEW_INSUFFICIENT_TRAILING_HISTORY")
                continue
            trailing_values.append(float(closes[position] / closes[start] - 1.0))
            window_returns = daily_returns[start + 1 : position + 1]
            if len(window_returns) != horizon or np.isnan(window_returns).any():
                volatility_values.append(np.nan)
                feature_statuses.append("NO_VIEW_TRAILING_RETURN_GAP")
            else:
                volatility_values.append(
                    float(np.std(window_returns, ddof=1) * np.sqrt(trading_days_per_year))
                )
                feature_statuses.append("PASS_POINT_IN_TIME_PRICE_FEATURE")
        result[f"trailing_total_return_{horizon}d"] = trailing_values
        result[f"realized_volatility_{horizon}d"] = volatility_values
        result[f"price_feature_{horizon}d_status"] = feature_statuses

    drawdown_values: list[float] = []
    drawdown_statuses: list[str] = []
    for position in positions:
        start = position - 60
        if position < 0 or start < 0:
            drawdown_values.append(np.nan)
            drawdown_statuses.append("NO_VIEW_INSUFFICIENT_TRAILING_HISTORY")
            continue
        window = closes[start : position + 1]
        running_peak = np.maximum.accumulate(window)
        drawdown_values.append(float(np.min(window / running_peak - 1.0)))
        drawdown_statuses.append("PASS_POINT_IN_TIME_PRICE_FEATURE")
    result["trailing_drawdown_60d"] = drawdown_values
    result["trailing_drawdown_60d_status"] = drawdown_statuses

    for horizon in return_horizons_market_days:
        values: list[float] = []
        maturities: list[pd.Timestamp | pd.NaT] = []
        target_statuses: list[str] = []
        for position in positions:
            target = position + int(horizon)
            if position < 0 or target >= len(closes):
                values.append(np.nan)
                maturities.append(pd.NaT)
                target_statuses.append("CENSORED_NO_VIEW")
            else:
                values.append(float(closes[target] / closes[position] - 1.0))
                maturities.append(dates[target])
                target_statuses.append("OBSERVED_PREDICTION_TARGET")
        result[f"target_return_{horizon}d"] = values
        result[f"target_return_{horizon}d_maturity_date"] = maturities
        result[f"target_return_{horizon}d_status"] = target_statuses

    future_min_returns: list[float] = []
    events: list[float] = []
    maturities = []
    target_statuses = []
    horizon = int(drawdown_horizon_market_days)
    for position in positions:
        target = position + horizon
        if position < 0 or target >= len(closes):
            future_min_returns.append(np.nan)
            events.append(np.nan)
            maturities.append(pd.NaT)
            target_statuses.append("CENSORED_NO_VIEW")
            continue
        future_path = closes[position + 1 : target + 1]
        minimum_return = float(np.min(future_path / closes[position] - 1.0))
        future_min_returns.append(minimum_return)
        events.append(float(minimum_return <= float(drawdown_threshold)))
        maturities.append(dates[target])
        target_statuses.append("OBSERVED_PREDICTION_TARGET")
    result["target_drawdown_20d_minimum_return"] = future_min_returns
    result["target_drawdown_20d_event"] = events
    result["target_drawdown_20d_maturity_date"] = maturities
    result["target_drawdown_20d_status"] = target_statuses
    result["h00300_values_read"] = True
    return result


def feature_columns_by_model(
    feature_groups: Mapping[str, Sequence[str]],
    model_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for model_id, specification in model_specs.items():
        columns: list[str] = []
        for group in specification["feature_groups"]:
            if group not in feature_groups:
                raise ValueError(f"模型 {model_id} 引用了未知特征组：{group}")
            columns.extend(str(column) for column in feature_groups[group])
        if len(columns) != len(set(columns)):
            raise ValueError(f"模型 {model_id} 的特征存在重复")
        result[model_id] = columns
    return result


def _target_contract(target_id: str, specification: Mapping[str, Any]) -> tuple[str, str]:
    if specification["target_type"] == "CONTINUOUS_RETURN":
        horizon = int(specification["horizon_market_days"])
        return f"target_return_{horizon}d", f"target_return_{horizon}d_maturity_date"
    if target_id == "20D_DRAWDOWN_PROBABILITY":
        return "target_drawdown_20d_event", "target_drawdown_20d_maturity_date"
    raise ValueError(f"未知预测目标：{target_id}")


def _standardize_training_and_current(
    training: pd.DataFrame,
    current: pd.Series,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train = training[list(columns)].apply(pd.to_numeric, errors="coerce")
    x_current = pd.to_numeric(current[list(columns)], errors="coerce")
    if x_train.isna().any().any() or x_current.isna().any():
        raise ValueError("固定模型特征存在缺失，协议禁止插补")
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0, ddof=1).replace(0.0, 1.0).fillna(1.0)
    return (
        ((x_train - means) / scales).to_numpy(dtype=float),
        ((x_current - means) / scales).to_numpy(dtype=float).reshape(1, -1),
    )


def walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    target_specs: Mapping[str, Mapping[str, Any]],
    model_specs: Mapping[str, Mapping[str, Any]],
    feature_groups: Mapping[str, Sequence[str]],
    walk_forward: Mapping[str, Any],
) -> pd.DataFrame:
    """使用严格到期标签执行扩展窗口月度样本外预测。"""

    data = frame.sort_values("origin").reset_index(drop=True).copy()
    data["origin"] = normalize_dates(data["origin"])
    model_features = feature_columns_by_model(feature_groups, model_specs)
    all_features = sorted(
        {column for columns in model_features.values() for column in columns}
    )
    _require_columns(
        data,
        ["origin", "condition_quality_status", *all_features],
        "预测建模面板",
    )
    minimum_training = int(walk_forward["minimum_matured_training_observations"])
    ridge_alpha = float(walk_forward["continuous_model"]["alpha"])
    binary_config = walk_forward["binary_model"]
    rows: list[dict[str, Any]] = []

    current_feature_ready = data[all_features].notna().all(axis=1)
    current_quality_ready = data["condition_quality_status"].eq(PASS_STATE_STATUS)
    prediction_indices = data.index[current_feature_ready & current_quality_ready]

    for target_id, target_specification in target_specs.items():
        target_column, maturity_column = _target_contract(
            target_id, target_specification
        )
        _require_columns(data, [target_column, maturity_column], "预测目标面板")
        is_binary = target_specification["target_type"] == "BINARY_PROBABILITY"
        for index in prediction_indices:
            current = data.loc[index]
            origin = pd.Timestamp(current["origin"])
            maturity = pd.to_datetime(data[maturity_column], errors="coerce")
            base_training_mask = (
                data["condition_quality_status"].eq(PASS_STATE_STATUS)
                & pd.to_numeric(data[target_column], errors="coerce").notna()
                & maturity.notna()
                & maturity.lt(origin)
            )
            actual = pd.to_numeric(
                pd.Series([current[target_column]]), errors="coerce"
            ).iloc[0]
            actual_value = np.nan if pd.isna(actual) else float(actual)
            actual_status = (
                "ACTUAL_OBSERVED" if not pd.isna(actual_value) else "ACTUAL_CENSORED"
            )

            for model_id, columns in model_features.items():
                feature_ready = (
                    data[columns].notna().all(axis=1)
                    if columns
                    else pd.Series(True, index=data.index)
                )
                training = data.loc[base_training_mask & feature_ready].copy()
                training_count = int(len(training))
                positive_events = (
                    int(pd.to_numeric(training[target_column]).sum())
                    if is_binary and training_count
                    else 0
                )
                negative_events = training_count - positive_events if is_binary else 0
                prediction = np.nan
                status = "NO_VIEW_MINIMUM_MATURED_TRAINING_NOT_MET"
                if training_count >= minimum_training:
                    y_train = pd.to_numeric(
                        training[target_column], errors="raise"
                    ).to_numpy(dtype=float)
                    if model_id == "HISTORICAL_MEAN":
                        if is_binary:
                            prediction = float(
                                (positive_events + 0.5) / (training_count + 1.0)
                            )
                        else:
                            prediction = float(np.mean(y_train))
                        status = "PREDICTION_GENERATED"
                    elif columns and not current[list(columns)].isna().any():
                        x_train, x_current = _standardize_training_and_current(
                            training, current, columns
                        )
                        if is_binary:
                            if (
                                positive_events
                                < int(binary_config["minimum_matured_positive_events"])
                                or negative_events
                                < int(binary_config["minimum_matured_negative_events"])
                            ):
                                status = "NO_VIEW_BINARY_CLASS_TRAINING_GATE"
                            else:
                                estimator = LogisticRegression(
                                    C=float(binary_config["c"]),
                                    fit_intercept=True,
                                    class_weight=None,
                                    solver="lbfgs",
                                    max_iter=int(
                                        binary_config["maximum_iterations"]
                                    ),
                                    random_state=0,
                                )
                                estimator.fit(x_train, y_train.astype(int))
                                prediction = float(
                                    estimator.predict_proba(x_current)[0, 1]
                                )
                                status = "PREDICTION_GENERATED"
                        else:
                            estimator = Ridge(
                                alpha=ridge_alpha,
                                fit_intercept=True,
                                solver="svd",
                            )
                            estimator.fit(x_train, y_train)
                            prediction = float(estimator.predict(x_current)[0])
                            status = "PREDICTION_GENERATED"

                max_maturity = (
                    pd.to_datetime(training[maturity_column]).max()
                    if training_count
                    else pd.NaT
                )
                if not pd.isna(max_maturity) and max_maturity >= origin:
                    raise ValueError("滚动训练集包含预测 origin 当日或之后才到期的标签")
                rows.append(
                    {
                        "origin": origin,
                        "target_id": target_id,
                        "target_type": target_specification["target_type"],
                        "horizon_market_days": int(
                            target_specification["horizon_market_days"]
                        ),
                        "model_id": model_id,
                        "prediction": prediction,
                        "actual": actual_value,
                        "actual_status": actual_status,
                        "target_maturity_date": current[maturity_column],
                        "training_observation_count": training_count,
                        "training_positive_event_count": positive_events,
                        "training_negative_event_count": negative_events,
                        "training_max_maturity_date": max_maturity,
                        "feature_count": int(len(columns)),
                        "prediction_status": (
                            status + "_" + actual_status
                            if status == "PREDICTION_GENERATED"
                            else status
                        ),
                        "portfolio_evaluation_allowed": False,
                        "model_position_target": "UNSET",
                    }
                )
    return pd.DataFrame(rows)


def _prediction_common_origins(
    scoped: pd.DataFrame, model_ids: Sequence[str]
) -> pd.DatetimeIndex:
    origin_sets: list[set[pd.Timestamp]] = []
    for model_id in model_ids:
        rows = scoped.loc[
            scoped["model_id"].eq(model_id)
            & scoped["prediction"].notna()
            & scoped["actual"].notna()
        ]
        origin_sets.append(set(pd.to_datetime(rows["origin"])))
    if not origin_sets:
        return pd.DatetimeIndex([])
    common = set.intersection(*origin_sets)
    return pd.DatetimeIndex(sorted(common))


def _reliability_status(
    actual: np.ndarray,
    origins: pd.Series,
    *,
    is_binary: bool,
    gate: Mapping[str, Any],
) -> str:
    observations = int(len(actual))
    years = int(pd.to_datetime(origins).dt.year.nunique())
    if observations < int(gate["minimum_observed_predictions"]):
        return "NO_VIEW_INSUFFICIENT_OOS_PREDICTIONS"
    if years < int(gate["minimum_distinct_calendar_years"]):
        return "NO_VIEW_INSUFFICIENT_OOS_YEAR_SPAN"
    if is_binary:
        positives = int(np.sum(actual == 1.0))
        negatives = int(np.sum(actual == 0.0))
        if positives < int(gate["binary_minimum_positive_events"]):
            return "NO_VIEW_INSUFFICIENT_OOS_DRAWDOWN_EVENTS"
        if negatives < int(gate["binary_minimum_negative_events"]):
            return "NO_VIEW_INSUFFICIENT_OOS_NON_EVENTS"
    return "PASS_OOS_EVALUATION_COVERAGE"


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    target_specs: Mapping[str, Mapping[str, Any]],
    model_ids: Sequence[str],
    sample_scopes: Mapping[str, Mapping[str, str]],
    reliability_gate: Mapping[str, Any],
) -> pd.DataFrame:
    """在所有模型共同可预测 origin 上计算固定样本外指标。"""

    metrics: list[dict[str, Any]] = []
    frame = predictions.copy()
    frame["origin"] = normalize_dates(frame["origin"])
    for scope_id, scope in sample_scopes.items():
        scoped_all = frame.loc[
            frame["origin"].between(
                pd.Timestamp(scope["start"]), pd.Timestamp(scope["end"])
            )
        ]
        for target_id, target_specification in target_specs.items():
            target_rows = scoped_all.loc[scoped_all["target_id"].eq(target_id)]
            common_origins = _prediction_common_origins(target_rows, model_ids)
            common = target_rows.loc[target_rows["origin"].isin(common_origins)]
            historical = common.loc[
                common["model_id"].eq("HISTORICAL_MEAN")
            ].set_index("origin")
            is_binary = target_specification["target_type"] == "BINARY_PROBABILITY"
            for model_id in model_ids:
                rows = common.loc[common["model_id"].eq(model_id)].sort_values(
                    "origin"
                )
                prediction = pd.to_numeric(rows["prediction"], errors="coerce").to_numpy(
                    dtype=float
                )
                actual = pd.to_numeric(rows["actual"], errors="coerce").to_numpy(
                    dtype=float
                )
                reliability = _reliability_status(
                    actual,
                    rows["origin"],
                    is_binary=is_binary,
                    gate=reliability_gate,
                )
                base = {
                    "sample_scope": scope_id,
                    "target_id": target_id,
                    "target_type": target_specification["target_type"],
                    "horizon_market_days": int(
                        target_specification["horizon_market_days"]
                    ),
                    "model_id": model_id,
                    "observed_prediction_count": int(len(rows)),
                    "distinct_calendar_years": int(
                        rows["origin"].dt.year.nunique()
                    ),
                    "first_prediction_origin": (
                        rows["origin"].min() if len(rows) else pd.NaT
                    ),
                    "last_prediction_origin": (
                        rows["origin"].max() if len(rows) else pd.NaT
                    ),
                    "positive_event_count": (
                        int(np.sum(actual == 1.0)) if is_binary else 0
                    ),
                    "negative_event_count": (
                        int(np.sum(actual == 0.0)) if is_binary else 0
                    ),
                    "reliability_status": reliability,
                    "common_origin_comparison": True,
                }
                if len(rows) == 0:
                    metrics.append(base)
                    continue
                if is_binary:
                    clipped = np.clip(prediction, 1.0e-9, 1.0 - 1.0e-9)
                    brier = float(np.mean((clipped - actual) ** 2))
                    log_loss = float(
                        -np.mean(
                            actual * np.log(clipped)
                            + (1.0 - actual) * np.log(1.0 - clipped)
                        )
                    )
                    if len(np.unique(actual)) == 2:
                        auc = float(roc_auc_score(actual, clipped))
                        average_precision = float(
                            average_precision_score(actual, clipped)
                        )
                    else:
                        auc = np.nan
                        average_precision = np.nan
                    historical_rows = historical.reindex(rows["origin"])
                    historical_prediction = pd.to_numeric(
                        historical_rows["prediction"], errors="coerce"
                    ).to_numpy(dtype=float)
                    historical_brier = float(
                        np.mean((historical_prediction - actual) ** 2)
                    )
                    base.update(
                        {
                            "loss_name": "BRIER",
                            "primary_loss": brier,
                            "brier_score": brier,
                            "log_loss": log_loss,
                            "roc_auc": auc,
                            "average_precision": average_precision,
                            "mean_predicted_probability": float(
                                np.mean(clipped)
                            ),
                            "observed_event_rate": float(np.mean(actual)),
                            "brier_skill_vs_historical_mean": (
                                float(1.0 - brier / historical_brier)
                                if historical_brier > 0
                                else np.nan
                            ),
                        }
                    )
                else:
                    errors = prediction - actual
                    mse = float(np.mean(errors**2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(errors)))
                    pearson = (
                        float(np.corrcoef(prediction, actual)[0, 1])
                        if len(rows) >= 2
                        and np.ptp(prediction) > 1.0e-15
                        and np.ptp(actual) > 1.0e-15
                        else np.nan
                    )
                    spearman = (
                        float(
                            pd.Series(prediction).corr(
                                pd.Series(actual), method="spearman"
                            )
                        )
                        if len(rows) >= 2
                        and np.ptp(prediction) > 1.0e-15
                        and np.ptp(actual) > 1.0e-15
                        else np.nan
                    )
                    sign_accuracy = float(
                        np.mean((prediction > 0) == (actual > 0))
                    )
                    historical_rows = historical.reindex(rows["origin"])
                    historical_prediction = pd.to_numeric(
                        historical_rows["prediction"], errors="coerce"
                    ).to_numpy(dtype=float)
                    historical_mse = float(
                        np.mean((historical_prediction - actual) ** 2)
                    )
                    base.update(
                        {
                            "loss_name": "MSE",
                            "primary_loss": mse,
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae,
                            "pearson_correlation": pearson,
                            "spearman_correlation": spearman,
                            "sign_accuracy": sign_accuracy,
                            "mean_prediction": float(np.mean(prediction)),
                            "mean_actual": float(np.mean(actual)),
                            "mean_error": float(np.mean(errors)),
                            "oos_r2_vs_historical_mean": (
                                float(1.0 - mse / historical_mse)
                                if historical_mse > 0
                                else np.nan
                            ),
                        }
                    )
                metrics.append(base)
    return pd.DataFrame(metrics)


def build_increment_results(
    metrics: pd.DataFrame,
    *,
    target_specs: Mapping[str, Mapping[str, Any]],
    primary_scope: str,
    module_comparisons: Mapping[str, Mapping[str, str]],
    final_model: str,
    final_baselines: Sequence[str],
    minimum_loss_improvement: float,
    continuous_auxiliary_gate: Mapping[str, float],
    binary_auxiliary_gate: Mapping[str, float],
) -> pd.DataFrame:
    """按冻结比较关系判断模块增量和最终模型相对基准的结果。"""

    scoped = metrics.loc[metrics["sample_scope"].eq(primary_scope)].copy()
    rows: list[dict[str, Any]] = []

    comparisons: list[tuple[str, str, str, str]] = []
    for comparison_id, specification in module_comparisons.items():
        comparisons.append(
            (
                "MODULE_INCREMENT",
                comparison_id,
                specification["candidate"],
                specification["reference"],
            )
        )
    for baseline in final_baselines:
        comparisons.append(
            (
                "FINAL_MODEL_VS_BASELINE",
                f"{final_model}_VS_{baseline}",
                final_model,
                baseline,
            )
        )

    for comparison_type, comparison_id, candidate_id, reference_id in comparisons:
        for target_id, target_specification in target_specs.items():
            candidate_rows = scoped.loc[
                scoped["target_id"].eq(target_id)
                & scoped["model_id"].eq(candidate_id)
            ]
            reference_rows = scoped.loc[
                scoped["target_id"].eq(target_id)
                & scoped["model_id"].eq(reference_id)
            ]
            if candidate_rows.empty or reference_rows.empty:
                rows.append(
                    {
                        "comparison_type": comparison_type,
                        "comparison_id": comparison_id,
                        "target_id": target_id,
                        "candidate_model": candidate_id,
                        "reference_model": reference_id,
                        "result": "NO_VIEW_MISSING_METRIC_ROW",
                    }
                )
                continue
            candidate = candidate_rows.iloc[0]
            reference = reference_rows.iloc[0]
            candidate_loss = candidate.get("primary_loss", np.nan)
            reference_loss = reference.get("primary_loss", np.nan)
            reliable = (
                candidate["reliability_status"] == "PASS_OOS_EVALUATION_COVERAGE"
                and reference["reliability_status"]
                == "PASS_OOS_EVALUATION_COVERAGE"
            )
            improvement = (
                float(1.0 - float(candidate_loss) / float(reference_loss))
                if pd.notna(candidate_loss)
                and pd.notna(reference_loss)
                and float(reference_loss) > 0
                else np.nan
            )
            if target_specification["target_type"] == "CONTINUOUS_RETURN":
                auxiliary_pass = (
                    pd.notna(candidate.get("spearman_correlation"))
                    and float(candidate["spearman_correlation"])
                    >= float(continuous_auxiliary_gate["minimum_spearman"])
                    and float(candidate["sign_accuracy"])
                    >= float(continuous_auxiliary_gate["minimum_sign_accuracy"])
                )
                auxiliary_metric = {
                    "spearman_correlation": candidate.get(
                        "spearman_correlation", np.nan
                    ),
                    "sign_accuracy": candidate.get("sign_accuracy", np.nan),
                }
            else:
                auxiliary_pass = (
                    pd.notna(candidate.get("roc_auc"))
                    and float(candidate["roc_auc"])
                    >= float(binary_auxiliary_gate["minimum_roc_auc"])
                )
                auxiliary_metric = {"roc_auc": candidate.get("roc_auc", np.nan)}
            if not reliable:
                result = "NO_VIEW_OOS_RELIABILITY_GATE"
            elif (
                pd.notna(improvement)
                and improvement >= float(minimum_loss_improvement)
                and auxiliary_pass
            ):
                result = "PASS_INCREMENTAL_INFORMATION"
            else:
                result = "FAIL_NO_INCREMENTAL_INFORMATION"
            rows.append(
                {
                    "comparison_type": comparison_type,
                    "comparison_id": comparison_id,
                    "target_id": target_id,
                    "target_type": target_specification["target_type"],
                    "candidate_model": candidate_id,
                    "reference_model": reference_id,
                    "candidate_primary_loss": candidate_loss,
                    "reference_primary_loss": reference_loss,
                    "relative_loss_improvement": improvement,
                    "minimum_required_improvement": float(
                        minimum_loss_improvement
                    ),
                    "candidate_reliability_status": candidate[
                        "reliability_status"
                    ],
                    "reference_reliability_status": reference[
                        "reliability_status"
                    ],
                    "auxiliary_gate_pass": bool(auxiliary_pass),
                    **auxiliary_metric,
                    "result": result,
                }
            )
    return pd.DataFrame(rows)


def validation_summary(
    increment_results: pd.DataFrame,
    *,
    module_comparison_ids: Sequence[str],
    final_model: str,
    final_baselines: Sequence[str],
    target_ids: Sequence[str],
) -> dict[str, Any]:
    module_statuses: dict[str, str] = {}
    for comparison_id in module_comparison_ids:
        rows = increment_results.loc[
            increment_results["comparison_type"].eq("MODULE_INCREMENT")
            & increment_results["comparison_id"].eq(comparison_id)
        ]
        passed_targets = sorted(
            rows.loc[
                rows["result"].eq("PASS_INCREMENTAL_INFORMATION"), "target_id"
            ].astype(str)
        )
        if len(rows) != len(target_ids) or rows["result"].str.startswith(
            "NO_VIEW"
        ).any():
            status = "NO_VIEW_OR_INCOMPLETE_INCREMENTAL_EVIDENCE"
        elif len(passed_targets) == len(target_ids):
            status = "PASS_ALL_THREE_TARGETS"
        elif passed_targets:
            status = "PARTIAL_PASS_NOT_ALL_TARGETS"
        else:
            status = "FAIL_ALL_TARGETS"
        module_statuses[comparison_id] = status

    baseline_statuses: dict[str, str] = {}
    for baseline in final_baselines:
        comparison_id = f"{final_model}_VS_{baseline}"
        rows = increment_results.loc[
            increment_results["comparison_type"].eq("FINAL_MODEL_VS_BASELINE")
            & increment_results["comparison_id"].eq(comparison_id)
        ]
        if len(rows) != len(target_ids) or rows["result"].str.startswith(
            "NO_VIEW"
        ).any():
            status = "NO_VIEW_OR_INCOMPLETE_BASELINE_COMPARISON"
        elif rows["result"].eq("PASS_INCREMENTAL_INFORMATION").all():
            status = "PASS_ALL_THREE_TARGETS"
        elif rows["result"].eq("PASS_INCREMENTAL_INFORMATION").any():
            status = "PARTIAL_PASS_NOT_ALL_TARGETS"
        else:
            status = "FAIL_ALL_TARGETS"
        baseline_statuses[baseline] = status

    validated = all(
        status == "PASS_ALL_THREE_TARGETS" for status in module_statuses.values()
    ) and all(
        status == "PASS_ALL_THREE_TARGETS" for status in baseline_statuses.values()
    )
    return {
        "module_increment_statuses": module_statuses,
        "final_model_vs_baseline_statuses": baseline_statuses,
        "structural_prediction_validated": validated,
        "validation_status": (
            "PASS_FULL_FROZEN_PREDICTIVE_VALIDATION"
            if validated
            else "FAIL_OR_NO_VIEW_FULL_FROZEN_PREDICTIVE_VALIDATION"
        ),
    }
