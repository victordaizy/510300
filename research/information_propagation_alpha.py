"""IF 与沪深300内部扩散领先 510300 的分钟级研究工具。

本模块只构造因子和条件收益统计，不生成实盘订单。所有滞后与远期收益都通过
精确时间戳连接计算，避免把午休或隔夜误当成相邻一分钟。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeadLagParameters:
    lookback_minutes: int = 3
    breadth_impulse_minutes: int = 5
    horizons_minutes: tuple[int, ...] = (1, 3, 5, 10, 15)
    leader_count: int = 50
    minimum_component_weight_coverage: float = 0.80
    beta: float = 1.0
    normalization_sessions: int = 20
    normalization_minimum_sessions: int = 10
    decile_count: int = 10
    residual_estimation_sessions: int = 20
    residual_minimum_sessions: int = 10
    execution_delay_minutes: int = 1

    def validate(self) -> None:
        if self.lookback_minutes < 1:
            raise ValueError("lookback_minutes 必须为正整数")
        if self.breadth_impulse_minutes < 1:
            raise ValueError("breadth_impulse_minutes 必须为正整数")
        if not self.horizons_minutes or min(self.horizons_minutes) < 1:
            raise ValueError("horizons_minutes 必须包含正整数")
        if self.leader_count < 1:
            raise ValueError("leader_count 必须为正整数")
        if not 0 < self.minimum_component_weight_coverage <= 1:
            raise ValueError("minimum_component_weight_coverage 必须位于 (0, 1]")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta 必须为有限正数")
        if self.normalization_sessions < 2:
            raise ValueError("normalization_sessions 至少为 2")
        if not 2 <= self.normalization_minimum_sessions <= self.normalization_sessions:
            raise ValueError("normalization_minimum_sessions 必须位于 [2, normalization_sessions]")
        if self.decile_count < 2:
            raise ValueError("decile_count 至少为 2")
        if not 2 <= self.residual_minimum_sessions <= self.residual_estimation_sessions:
            raise ValueError("残差估计最少交易日必须位于 [2, residual_estimation_sessions]")
        if self.execution_delay_minutes < 1:
            raise ValueError("execution_delay_minutes 必须至少为1，禁止信号同分钟成交")


def _require_columns(data: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{label}缺少字段：{missing}")


def _normalize_timestamp(data: pd.DataFrame, timestamp_column: str, label: str) -> pd.DataFrame:
    output = data.copy()
    output[timestamp_column] = pd.to_datetime(
        output[timestamp_column], errors="coerce"
    ).astype("datetime64[ns]")
    if output[timestamp_column].isna().any():
        raise ValueError(f"{label}存在无法解析的时间戳")
    if output[timestamp_column].dt.tz is not None:
        output[timestamp_column] = output[timestamp_column].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    output["session_date"] = output[timestamp_column].dt.normalize()
    return output


def _validate_prices(data: pd.DataFrame, price_column: str, label: str) -> pd.DataFrame:
    output = data.copy()
    output[price_column] = pd.to_numeric(output[price_column], errors="coerce")
    if output[price_column].isna().any() or (output[price_column] <= 0).any():
        raise ValueError(f"{label}存在无效价格")
    return output


def _exact_lag_return(
    data: pd.DataFrame,
    timestamp_column: str,
    price_column: str,
    minutes: int,
    group_columns: tuple[str, ...] = (),
) -> pd.Series:
    """计算精确相隔 minutes 分钟的简单收益；午休、隔夜和缺柱均返回 NaN。"""

    key_columns = [*group_columns, timestamp_column]
    if data.duplicated(key_columns).any():
        raise ValueError(f"收益输入在键 {key_columns} 上存在重复")
    left = data[key_columns].copy()
    left["_target_time"] = left[timestamp_column] - pd.Timedelta(minutes=minutes)
    right = data[[*group_columns, timestamp_column, price_column]].rename(
        columns={timestamp_column: "_target_time", price_column: "_lag_price"}
    )
    merged = left.merge(right, on=[*group_columns, "_target_time"], how="left", sort=False)
    values = data[price_column].to_numpy(dtype=float) / merged["_lag_price"].to_numpy(dtype=float) - 1.0
    return pd.Series(values, index=data.index, dtype=float)


def _exact_future_return(
    data: pd.DataFrame,
    timestamp_column: str,
    price_column: str,
    minutes: int,
) -> pd.Series:
    if data[timestamp_column].duplicated().any():
        raise ValueError("ETF 时间戳存在重复")
    left = data[[timestamp_column]].copy()
    left["_target_time"] = left[timestamp_column] + pd.Timedelta(minutes=minutes)
    right = data[[timestamp_column, price_column]].rename(
        columns={timestamp_column: "_target_time", price_column: "_future_price"}
    )
    merged = left.merge(right, on="_target_time", how="left", sort=False)
    values = merged["_future_price"].to_numpy(dtype=float) / data[price_column].to_numpy(dtype=float) - 1.0
    return pd.Series(values, index=data.index, dtype=float)


def _exact_future_value(
    data: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    minutes: int,
) -> pd.Series:
    """取得精确未来时点的字段值；目标分钟不存在时返回 NaN。"""

    left = data[[timestamp_column]].copy()
    left["_target_time"] = left[timestamp_column] + pd.Timedelta(minutes=minutes)
    right = data[[timestamp_column, value_column]].rename(
        columns={timestamp_column: "_target_time", value_column: "_future_value"}
    )
    merged = left.merge(right, on="_target_time", how="left", sort=False)
    return pd.Series(merged["_future_value"].to_numpy(dtype=float), index=data.index, dtype=float)


def _exact_execution_return(
    data: pd.DataFrame,
    entry_column: str,
    exit_column: str,
    delay_minutes: int,
    holding_minutes: int,
) -> pd.Series:
    """信号结束后延迟入场，并从入场时点持有精确分钟数。"""

    entry = _exact_future_value(data, "trade_time", entry_column, delay_minutes)
    exit_price = _exact_future_value(
        data, "trade_time", exit_column, delay_minutes + holding_minutes
    )
    values = exit_price.to_numpy(dtype=float) / entry.to_numpy(dtype=float) - 1.0
    return pd.Series(values, index=data.index, dtype=float)


def select_causal_futures_contract(
    futures: pd.DataFrame,
    timestamp_column: str = "trade_time",
    contract_column: str = "contract",
    volume_column: str = "volume",
) -> pd.DataFrame:
    """按前一交易日总成交量选择当日 IF 合约，且整日不换约。

    如果输入没有 contract 字段，则要求每个时间戳唯一，并原样返回。该情形适合
    已由上游按同一规则构造且保留换月审计信息的分钟序列。
    """

    _require_columns(futures, [timestamp_column], "IF 分钟数据")
    data = _normalize_timestamp(futures, timestamp_column, "IF 分钟数据")
    if contract_column not in data.columns:
        if data[timestamp_column].duplicated().any():
            raise ValueError("没有 contract 字段时，IF 每个时间戳必须唯一")
        return data.sort_values(timestamp_column).reset_index(drop=True)

    _require_columns(data, [volume_column], "含多合约的 IF 分钟数据")
    data[volume_column] = pd.to_numeric(data[volume_column], errors="coerce")
    if data[volume_column].isna().any() or (data[volume_column] < 0).any():
        raise ValueError("IF 分钟数据存在无效成交量")
    if data.duplicated([contract_column, timestamp_column]).any():
        raise ValueError("IF 合约与时间戳组合存在重复")

    daily = (
        data.groupby(["session_date", contract_column], as_index=False)[volume_column]
        .sum()
        .sort_values(["session_date", volume_column, contract_column], ascending=[True, False, True])
    )
    winners = daily.drop_duplicates("session_date", keep="first")[["session_date", contract_column]]
    dates = pd.DataFrame({"session_date": sorted(data["session_date"].unique())})
    previous = winners.rename(
        columns={"session_date": "selection_source_date", contract_column: "selected_contract"}
    ).sort_values("selection_source_date")
    dates["selection_cutoff"] = dates["session_date"] - pd.Timedelta(nanoseconds=1)
    mapping = pd.merge_asof(
        dates.sort_values("selection_cutoff"),
        previous,
        left_on="selection_cutoff",
        right_on="selection_source_date",
        direction="backward",
    )
    selected = data.merge(mapping[["session_date", "selected_contract", "selection_source_date"]], on="session_date", how="left")
    selected = selected.loc[selected[contract_column] == selected["selected_contract"]].copy()
    return selected.sort_values(timestamp_column).reset_index(drop=True)


def _causal_weight_map(
    session_dates: pd.Series,
    weights: pd.DataFrame,
    leader_count: int,
    symbol_column: str,
    weight_date_column: str,
    weight_column: str,
) -> pd.DataFrame:
    _require_columns(weights, [symbol_column, weight_date_column, weight_column], "历史权重")
    snapshots = weights[[symbol_column, weight_date_column, weight_column]].copy()
    snapshots[weight_date_column] = (
        pd.to_datetime(snapshots[weight_date_column], errors="coerce")
        .astype("datetime64[ns]")
        .dt.normalize()
    )
    snapshots[weight_column] = pd.to_numeric(snapshots[weight_column], errors="coerce")
    if snapshots[[weight_date_column, weight_column]].isna().any().any():
        raise ValueError("历史权重存在无法解析的日期或权重")
    if (snapshots[weight_column] <= 0).any():
        raise ValueError("历史权重必须为正数")
    if snapshots.duplicated([weight_date_column, symbol_column]).any():
        raise ValueError("历史权重在日期与成分代码上存在重复")

    snapshot_dates = pd.DataFrame(
        {weight_date_column: sorted(snapshots[weight_date_column].unique())}
    )
    dates = pd.DataFrame({"session_date": sorted(pd.to_datetime(session_dates).dt.normalize().unique())})
    dates["weight_cutoff"] = dates["session_date"] - pd.Timedelta(nanoseconds=1)
    date_map = pd.merge_asof(
        dates.sort_values("weight_cutoff"),
        snapshot_dates,
        left_on="weight_cutoff",
        right_on=weight_date_column,
        direction="backward",
    ).dropna(subset=[weight_date_column])
    leaders = (
        snapshots.sort_values([weight_date_column, weight_column, symbol_column], ascending=[True, False, True])
        .groupby(weight_date_column, group_keys=False)
        .head(leader_count)
    )
    mapped = date_map[["session_date", weight_date_column]].merge(
        leaders, on=weight_date_column, how="left", validate="many_to_many"
    )
    mapped["leader_weight"] = mapped[weight_column] / mapped.groupby("session_date")[weight_column].transform("sum")
    return mapped[["session_date", weight_date_column, symbol_column, "leader_weight"]]


def build_intraday_breadth(
    components: pd.DataFrame,
    weights: pd.DataFrame,
    parameters: LeadLagParameters,
    timestamp_column: str = "trade_time",
    symbol_column: str = "con_code",
    price_column: str = "close",
    weight_date_column: str = "trade_date",
    weight_column: str = "weight",
) -> pd.DataFrame:
    parameters.validate()
    _require_columns(components, [timestamp_column, symbol_column, price_column], "成分股分钟数据")
    data = _normalize_timestamp(components, timestamp_column, "成分股分钟数据")
    data = _validate_prices(data, price_column, "成分股分钟数据")
    if data.duplicated([symbol_column, timestamp_column]).any():
        raise ValueError("成分股代码与时间戳组合存在重复")

    leaders = _causal_weight_map(
        data["session_date"], weights, parameters.leader_count,
        symbol_column, weight_date_column, weight_column,
    )
    data = data.merge(leaders, on=["session_date", symbol_column], how="inner", validate="many_to_one")
    data["component_return"] = _exact_lag_return(
        data, timestamp_column, price_column, parameters.lookback_minutes, (symbol_column,)
    )
    data["component_available"] = data["component_return"].notna().astype(int)
    data["positive_component"] = data["component_return"].gt(0).astype(int).where(
        data["component_return"].notna(), 0
    )
    data["available_weight"] = data["leader_weight"].where(data["component_return"].notna(), 0.0)
    data["weighted_return"] = data["component_return"].fillna(0.0) * data["leader_weight"]
    data["positive_weight"] = (
        data["component_return"].gt(0).astype(float) * data["leader_weight"]
    ).where(data["component_return"].notna(), 0.0)
    has_vwap_inputs = {"amount", "volume"}.issubset(data.columns)
    if has_vwap_inputs:
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
        data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
        data["session_amount"] = data.groupby(["session_date", symbol_column])["amount"].cumsum()
        data["session_volume"] = data.groupby(["session_date", symbol_column])["volume"].cumsum()
        data["session_vwap"] = data["session_amount"] / data["session_volume"].replace(0.0, np.nan)
        data["vwap_available"] = data["session_vwap"].notna().astype(int)
        data["above_vwap_component"] = (data[price_column] > data["session_vwap"]).astype(int).where(
            data["session_vwap"].notna(), 0
        )
        data["above_vwap_weight"] = (
            data["above_vwap_component"] * data["leader_weight"]
        ).where(data["session_vwap"].notna(), 0.0)
        data["vwap_available_weight"] = data["leader_weight"].where(
            data["session_vwap"].notna(), 0.0
        )
    grouped = data.groupby(timestamp_column, as_index=False).agg(
        session_date=("session_date", "first"),
        weight_snapshot_date=(weight_date_column, "first"),
        leader_weight_coverage=("available_weight", "sum"),
        weighted_return_sum=("weighted_return", "sum"),
        positive_weight_sum=("positive_weight", "sum"),
        available_component_count=("component_return", "count"),
        positive_component_count=("positive_component", "sum"),
    )
    if has_vwap_inputs:
        vwap_grouped = data.groupby(timestamp_column, as_index=False).agg(
            vwap_available_component_count=("vwap_available", "sum"),
            above_vwap_component_count=("above_vwap_component", "sum"),
            vwap_available_weight=("vwap_available_weight", "sum"),
            above_vwap_weight_sum=("above_vwap_weight", "sum"),
        )
        grouped = grouped.merge(vwap_grouped, on=timestamp_column, how="left", validate="one_to_one")
    valid = grouped["leader_weight_coverage"] >= parameters.minimum_component_weight_coverage
    minimum_component_count = int(np.ceil(parameters.leader_count * parameters.minimum_component_weight_coverage))
    count_valid = grouped["available_component_count"] >= minimum_component_count
    grouped["leader_return"] = np.where(
        valid,
        grouped["weighted_return_sum"] / grouped["leader_weight_coverage"],
        np.nan,
    )
    grouped["top50_weighted_breadth"] = np.where(
        valid,
        grouped["positive_weight_sum"] / grouped["leader_weight_coverage"],
        np.nan,
    )
    grouped["top50_breadth"] = np.where(
        count_valid,
        grouped["positive_component_count"] / grouped["available_component_count"],
        np.nan,
    )
    if has_vwap_inputs:
        vwap_count_valid = grouped["vwap_available_component_count"] >= minimum_component_count
        vwap_weight_valid = grouped["vwap_available_weight"] >= parameters.minimum_component_weight_coverage
        grouped["top50_above_vwap_breadth"] = np.where(
            vwap_count_valid,
            grouped["above_vwap_component_count"] / grouped["vwap_available_component_count"],
            np.nan,
        )
        grouped["top50_weighted_above_vwap_breadth"] = np.where(
            vwap_weight_valid,
            grouped["above_vwap_weight_sum"] / grouped["vwap_available_weight"],
            np.nan,
        )
    else:
        grouped["top50_above_vwap_breadth"] = np.nan
        grouped["top50_weighted_above_vwap_breadth"] = np.nan
    for breadth_column in [
        "top50_breadth", "top50_weighted_breadth",
        "top50_above_vwap_breadth", "top50_weighted_above_vwap_breadth",
    ]:
        breadth_base = grouped[[timestamp_column, breadth_column]].copy()
        grouped[f"{breadth_column}_impulse_{parameters.breadth_impulse_minutes}m"] = (
            grouped[breadth_column] - _exact_lag_return_on_level(
                breadth_base,
                timestamp_column,
                breadth_column,
                parameters.breadth_impulse_minutes,
            )
        )
    grouped["breadth_fraction"] = grouped["top50_weighted_breadth"]
    grouped["breadth_impulse"] = grouped[
        f"top50_breadth_impulse_{parameters.breadth_impulse_minutes}m"
    ]
    return grouped.sort_values(timestamp_column).reset_index(drop=True)


def _exact_lag_return_on_level(
    data: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    minutes: int,
) -> pd.Series:
    """返回精确滞后水平而非收益，供比例型 breadth 做差。"""

    left = data[[timestamp_column]].copy()
    left["_target_time"] = left[timestamp_column] - pd.Timedelta(minutes=minutes)
    right = data[[timestamp_column, value_column]].rename(
        columns={timestamp_column: "_target_time", value_column: "_lag_value"}
    )
    merged = left.merge(right, on="_target_time", how="left", sort=False)
    return pd.Series(merged["_lag_value"].to_numpy(dtype=float), index=data.index, dtype=float)


def _causal_time_of_day_zscore(
    data: pd.DataFrame,
    value_column: str,
    timestamp_column: str,
    sessions: int,
    minimum_sessions: int,
) -> pd.Series:
    output = pd.Series(np.nan, index=data.index, dtype=float)
    minute_key = data[timestamp_column].dt.strftime("%H:%M")
    for _, index in minute_key.groupby(minute_key).groups.items():
        ordered = data.loc[index].sort_values(timestamp_column)
        history = ordered[value_column].shift(1)
        mean = history.rolling(sessions, min_periods=minimum_sessions).mean()
        std = history.rolling(sessions, min_periods=minimum_sessions).std(ddof=0)
        zscore = (ordered[value_column] - mean) / std.replace(0.0, np.nan)
        output.loc[ordered.index] = zscore
    return output


def _causal_if_residual(
    data: pd.DataFrame,
    sessions: int,
    minimum_sessions: int,
) -> pd.DataFrame:
    """只用当前交易日前的窗口估计 IF 对指数和 ETF 的线性暴露。"""

    result = pd.DataFrame(
        {
            "if_unique_residual": np.nan,
            "if_residual_intercept": np.nan,
            "if_residual_beta_index": np.nan,
            "if_residual_beta_etf": np.nan,
        },
        index=data.index,
    )
    dates = sorted(data["session_date"].dropna().unique())
    for position, session_date in enumerate(dates):
        prior_dates = dates[max(0, position - sessions):position]
        if len(prior_dates) < minimum_sessions:
            continue
        history = data.loc[
            data["session_date"].isin(prior_dates),
            ["if_return", "index_return", "etf_return"],
        ].dropna()
        if history.empty or history["if_return"].size < 30:
            continue
        design = np.column_stack(
            [
                np.ones(len(history), dtype=float),
                history["index_return"].to_numpy(dtype=float),
                history["etf_return"].to_numpy(dtype=float),
            ]
        )
        target = history["if_return"].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        current_index = data.index[data["session_date"] == session_date]
        current = data.loc[current_index, ["if_return", "index_return", "etf_return"]]
        valid = current.notna().all(axis=1)
        prediction = (
            coefficients[0]
            + coefficients[1] * current.loc[valid, "index_return"]
            + coefficients[2] * current.loc[valid, "etf_return"]
        )
        result.loc[current_index, "if_residual_intercept"] = coefficients[0]
        result.loc[current_index, "if_residual_beta_index"] = coefficients[1]
        result.loc[current_index, "if_residual_beta_etf"] = coefficients[2]
        result.loc[current.loc[valid].index, "if_unique_residual"] = (
            current.loc[valid, "if_return"] - prediction
        )
    return result


def build_information_propagation_features(
    etf: pd.DataFrame,
    futures: pd.DataFrame,
    components: pd.DataFrame,
    weights: pd.DataFrame,
    parameters: LeadLagParameters | None = None,
    index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    params = parameters or LeadLagParameters()
    params.validate()

    _require_columns(etf, ["trade_time", "close"], "510300 分钟数据")
    etf_data = _normalize_timestamp(etf, "trade_time", "510300 分钟数据")
    etf_data = _validate_prices(etf_data, "close", "510300 分钟数据")
    if etf_data["trade_time"].duplicated().any():
        raise ValueError("510300 分钟数据存在重复时间戳")
    etf_data = etf_data.sort_values("trade_time").reset_index(drop=True)
    etf_data["etf_return"] = _exact_lag_return(
        etf_data, "trade_time", "close", params.lookback_minutes
    )
    if "open" not in etf_data.columns:
        etf_data["open"] = etf_data["close"]
    etf_data = _validate_prices(etf_data, "open", "510300 分钟数据")
    if {"amount", "vol"}.issubset(etf_data.columns):
        amount = pd.to_numeric(etf_data["amount"], errors="coerce")
        volume = pd.to_numeric(etf_data["vol"], errors="coerce")
        etf_data["minute_vwap"] = amount / volume.replace(0.0, np.nan)
    elif {"amount", "volume"}.issubset(etf_data.columns):
        amount = pd.to_numeric(etf_data["amount"], errors="coerce")
        volume = pd.to_numeric(etf_data["volume"], errors="coerce")
        etf_data["minute_vwap"] = amount / volume.replace(0.0, np.nan)
    else:
        etf_data["minute_vwap"] = etf_data["open"]

    if index is None:
        index_data = etf_data[["trade_time", "close"]].rename(columns={"close": "index_close"})
    else:
        _require_columns(index, ["trade_time", "close"], "000300 指数分钟数据")
        normalized_index = _normalize_timestamp(index, "trade_time", "000300 指数分钟数据")
        normalized_index = _validate_prices(normalized_index, "close", "000300 指数分钟数据")
        if normalized_index["trade_time"].duplicated().any():
            raise ValueError("000300 指数分钟数据存在重复时间戳")
        index_data = normalized_index[["trade_time", "close"]].rename(
            columns={"close": "index_close"}
        )
    index_data = index_data.sort_values("trade_time").reset_index(drop=True)
    index_data["index_return"] = _exact_lag_return(
        index_data, "trade_time", "index_close", params.lookback_minutes
    )

    futures_data = select_causal_futures_contract(futures)
    _require_columns(futures_data, ["trade_time", "close"], "IF 分钟数据")
    futures_data = _validate_prices(futures_data, "close", "IF 分钟数据")
    futures_data["if_return"] = _exact_lag_return(
        futures_data, "trade_time", "close", params.lookback_minutes,
        ("contract",) if "contract" in futures_data.columns else (),
    )
    futures_columns = ["trade_time", "if_return"]
    for optional in ["contract", "selection_source_date"]:
        if optional in futures_data.columns:
            futures_columns.append(optional)

    breadth = build_intraday_breadth(components, weights, params)
    output = etf_data[
        ["trade_time", "session_date", "open", "close", "minute_vwap", "etf_return"]
    ].merge(
        index_data[["trade_time", "index_close", "index_return"]],
        on="trade_time", how="left", validate="one_to_one",
    ).merge(
        futures_data[futures_columns], on="trade_time", how="left", validate="one_to_one"
    ).merge(
        breadth.drop(columns="session_date"), on="trade_time", how="left", validate="one_to_one"
    )
    suffix = f"{params.lookback_minutes}m"
    output[f"if_lead_etf_{suffix}"] = output["if_return"] - params.beta * output["etf_return"]
    output[f"if_lead_index_{suffix}"] = output["if_return"] - params.beta * output["index_return"]
    output[f"top50_leader_lead_etf_{suffix}"] = output["leader_return"] - output["etf_return"]
    output[f"top50_leader_lead_index_{suffix}"] = output["leader_return"] - output["index_return"]
    output["if_lead"] = output[f"if_lead_etf_{suffix}"]
    output["leader_lead"] = output[f"top50_leader_lead_etf_{suffix}"]
    residual = _causal_if_residual(
        output, params.residual_estimation_sessions, params.residual_minimum_sessions
    )
    output = pd.concat([output, residual], axis=1)
    output["if_lead_z"] = _causal_time_of_day_zscore(
        output, "if_lead", "trade_time",
        params.normalization_sessions, params.normalization_minimum_sessions,
    )
    breadth_impulse_column = f"top50_breadth_impulse_{params.breadth_impulse_minutes}m"
    output["top50_breadth_impulse_z"] = _causal_time_of_day_zscore(
        output, breadth_impulse_column, "trade_time",
        params.normalization_sessions, params.normalization_minimum_sessions,
    )
    output["breadth_impulse_z"] = output["top50_breadth_impulse_z"]
    output["primary_alpha_score"] = output["if_lead_z"] + output["top50_breadth_impulse_z"]
    output["alpha_score"] = output["primary_alpha_score"]
    for horizon in params.horizons_minutes:
        output[f"etf_markout_close_{horizon}m"] = _exact_future_return(
            etf_data, "trade_time", "close", horizon
        )
        output[f"etf_forward_{horizon}m"] = output[f"etf_markout_close_{horizon}m"]
        output[f"etf_execution_next_open_{horizon}m"] = _exact_execution_return(
            etf_data, "open", "close", params.execution_delay_minutes, horizon
        )
        output[f"etf_execution_next_vwap_{horizon}m"] = _exact_execution_return(
            etf_data, "minute_vwap", "close", params.execution_delay_minutes, horizon
        )
    return output.sort_values("trade_time").reset_index(drop=True)


def build_no_if_information_propagation_features(
    etf: pd.DataFrame,
    index: pd.DataFrame,
    breadth: pd.DataFrame,
    parameters: LeadLagParameters | None = None,
) -> pd.DataFrame:
    """构造000300与Top50内部扩散领先510300的独立无IF实验特征。"""

    params = parameters or LeadLagParameters()
    params.validate()
    _require_columns(etf, ["trade_time", "open", "close"], "510300 分钟数据")
    etf_data = _normalize_timestamp(etf, "trade_time", "510300 分钟数据")
    etf_data = _validate_prices(etf_data, "open", "510300 分钟数据")
    etf_data = _validate_prices(etf_data, "close", "510300 分钟数据")
    if etf_data["trade_time"].duplicated().any():
        raise ValueError("510300 分钟数据存在重复时间戳")
    etf_data = etf_data.sort_values("trade_time").reset_index(drop=True)
    etf_data["etf_return"] = _exact_lag_return(
        etf_data, "trade_time", "close", params.lookback_minutes
    )
    if {"amount", "vol"}.issubset(etf_data.columns):
        amount = pd.to_numeric(etf_data["amount"], errors="coerce")
        volume = pd.to_numeric(etf_data["vol"], errors="coerce")
        etf_data["minute_vwap"] = amount / volume.replace(0.0, np.nan)
    else:
        etf_data["minute_vwap"] = etf_data["open"]

    _require_columns(index, ["trade_time", "close"], "000300 指数分钟数据")
    index_data = _normalize_timestamp(index, "trade_time", "000300 指数分钟数据")
    index_data = _validate_prices(index_data, "close", "000300 指数分钟数据")
    if index_data["trade_time"].duplicated().any():
        raise ValueError("000300 指数分钟数据存在重复时间戳")
    index_data = index_data.sort_values("trade_time").reset_index(drop=True)
    index_data["index_return"] = _exact_lag_return(
        index_data, "trade_time", "close", params.lookback_minutes
    )
    index_data = index_data.rename(columns={"close": "index_close"})

    impulse_column = f"top50_breadth_impulse_{params.breadth_impulse_minutes}m"
    _require_columns(
        breadth,
        ["trade_time", "leader_return", "top50_breadth", impulse_column],
        "Top50Breadth分钟数据",
    )
    breadth_data = _normalize_timestamp(breadth, "trade_time", "Top50Breadth分钟数据")
    if breadth_data["trade_time"].duplicated().any():
        raise ValueError("Top50Breadth分钟数据存在重复时间戳")
    breadth_columns = [
        column for column in breadth_data.columns
        if column not in {"session_date"}
    ]
    output = etf_data[
        ["trade_time", "session_date", "open", "close", "minute_vwap", "etf_return"]
    ].merge(
        index_data[["trade_time", "index_close", "index_return"]],
        on="trade_time", how="left", validate="one_to_one",
    ).merge(
        breadth_data[breadth_columns], on="trade_time", how="left", validate="one_to_one"
    )
    suffix = f"{params.lookback_minutes}m"
    output[f"index_lead_etf_{suffix}"] = (
        output["index_return"] - params.beta * output["etf_return"]
    )
    output[f"top50_leader_lead_etf_{suffix}"] = (
        output["leader_return"] - output["etf_return"]
    )
    output["index_lead_etf_z"] = _causal_time_of_day_zscore(
        output,
        f"index_lead_etf_{suffix}",
        "trade_time",
        params.normalization_sessions,
        params.normalization_minimum_sessions,
    )
    output["top50_breadth_impulse_z"] = _causal_time_of_day_zscore(
        output,
        impulse_column,
        "trade_time",
        params.normalization_sessions,
        params.normalization_minimum_sessions,
    )
    output["no_if_primary_alpha_score"] = (
        output["index_lead_etf_z"] + output["top50_breadth_impulse_z"]
    )
    for horizon in params.horizons_minutes:
        output[f"etf_markout_close_{horizon}m"] = _exact_future_return(
            etf_data, "trade_time", "close", horizon
        )
        output[f"etf_execution_next_open_{horizon}m"] = _exact_execution_return(
            etf_data, "open", "close", params.execution_delay_minutes, horizon
        )
        output[f"etf_execution_next_vwap_{horizon}m"] = _exact_execution_return(
            etf_data, "minute_vwap", "close", params.execution_delay_minutes, horizon
        )
    return output.sort_values("trade_time").reset_index(drop=True)


def fit_quantile_edges(signal: pd.Series, quantile_count: int = 10) -> np.ndarray:
    clean = pd.to_numeric(signal, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < quantile_count * 2:
        raise ValueError("开发期有效信号不足以稳定估计分位边界")
    probabilities = np.linspace(0.0, 1.0, quantile_count + 1)
    edges = np.quantile(clean, probabilities)
    interior = np.unique(edges[1:-1])
    if len(interior) != quantile_count - 1:
        raise ValueError("开发期信号重复值过多，无法形成预定数量的分位组")
    return np.concatenate(([-np.inf], interior, [np.inf]))


def summarize_signal_deciles(
    features: pd.DataFrame,
    signal_column: str,
    horizons_minutes: Iterable[int],
    development_start: str,
    development_end: str,
    evaluation_periods: dict[str, tuple[str, str]],
    quantile_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用开发期固定边界汇总各时间段分位收益和年度极端组差。"""

    _require_columns(features, ["trade_time", signal_column], "信息传播特征")
    data = features.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    development_mask = data["trade_time"].between(
        pd.Timestamp(development_start), pd.Timestamp(development_end) + pd.Timedelta(days=1),
        inclusive="left",
    )
    edges = fit_quantile_edges(data.loc[development_mask, signal_column], quantile_count)
    labels = list(range(1, quantile_count + 1))
    data["signal_bucket"] = pd.cut(
        data[signal_column], bins=edges, labels=labels, include_lowest=True
    ).astype("Int64")

    summaries: list[pd.DataFrame] = []
    yearly: list[dict[str, float | int | str]] = []
    for period_name, (start, end) in evaluation_periods.items():
        mask = data["trade_time"].between(
            pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), inclusive="left"
        )
        period = data.loc[mask].copy()
        for horizon in horizons_minutes:
            outcome = f"etf_forward_{horizon}m"
            _require_columns(period, [outcome], "信息传播特征")
            valid = period.dropna(subset=["signal_bucket", outcome])
            if valid.empty:
                continue
            summary = valid.groupby("signal_bucket", observed=True)[outcome].agg(
                observation_count="count",
                mean_return="mean",
                median_return="median",
                win_rate=lambda values: float((values > 0).mean()),
                standard_deviation="std",
            ).reset_index()
            summary["period"] = period_name
            summary["horizon_minutes"] = horizon
            summary["mean_return_bps"] = summary["mean_return"] * 10_000
            summary["median_return_bps"] = summary["median_return"] * 10_000
            summaries.append(summary)

            valid["year"] = valid["trade_time"].dt.year
            for year, year_data in valid.groupby("year"):
                low = year_data.loc[year_data["signal_bucket"] == 1, outcome]
                high = year_data.loc[year_data["signal_bucket"] == quantile_count, outcome]
                if low.empty or high.empty:
                    continue
                yearly.append(
                    {
                        "period": period_name,
                        "year": int(year),
                        "horizon_minutes": horizon,
                        "low_bucket_count": int(len(low)),
                        "high_bucket_count": int(len(high)),
                        "low_bucket_mean_bps": float(low.mean() * 10_000),
                        "high_bucket_mean_bps": float(high.mean() * 10_000),
                        "high_minus_low_bps": float((high.mean() - low.mean()) * 10_000),
                    }
                )
    deciles = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return deciles, pd.DataFrame(yearly)


def summarize_outcome_deciles(
    features: pd.DataFrame,
    signal_column: str,
    outcome_columns: Iterable[str],
    development_start: str,
    development_end: str,
    evaluation_periods: dict[str, tuple[str, str]],
    quantile_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对任意统计、执行或成本后结果使用同一组开发期分位边界。"""

    _require_columns(features, ["trade_time", signal_column, *outcome_columns], "信息传播特征")
    data = features.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    development_mask = data["trade_time"].between(
        pd.Timestamp(development_start),
        pd.Timestamp(development_end) + pd.Timedelta(days=1),
        inclusive="left",
    )
    edges = fit_quantile_edges(data.loc[development_mask, signal_column], quantile_count)
    data["signal_bucket"] = pd.cut(
        data[signal_column],
        bins=edges,
        labels=list(range(1, quantile_count + 1)),
        include_lowest=True,
    ).astype("Int64")

    summaries: list[pd.DataFrame] = []
    yearly_rows: list[dict[str, float | int | str]] = []
    for period_name, (start, end) in evaluation_periods.items():
        period_mask = data["trade_time"].between(
            pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), inclusive="left"
        )
        for outcome in outcome_columns:
            valid = data.loc[period_mask].dropna(subset=["signal_bucket", outcome]).copy()
            if valid.empty:
                continue
            summary = valid.groupby("signal_bucket", observed=True)[outcome].agg(
                observation_count="count",
                mean_return="mean",
                median_return="median",
                win_rate=lambda values: float((values > 0).mean()),
                standard_deviation="std",
            ).reset_index()
            summary["period"] = period_name
            summary["outcome"] = outcome
            summary["mean_return_bps"] = summary["mean_return"] * 10_000
            summary["median_return_bps"] = summary["median_return"] * 10_000
            summaries.append(summary)
            valid["year"] = valid["trade_time"].dt.year
            for year, year_data in valid.groupby("year"):
                low = year_data.loc[year_data["signal_bucket"] == 1, outcome]
                high = year_data.loc[year_data["signal_bucket"] == quantile_count, outcome]
                if low.empty or high.empty:
                    continue
                yearly_rows.append(
                    {
                        "period": period_name,
                        "year": int(year),
                        "outcome": outcome,
                        "low_bucket_count": int(len(low)),
                        "high_bucket_count": int(len(high)),
                        "low_bucket_mean_bps": float(low.mean() * 10_000),
                        "high_bucket_mean_bps": float(high.mean() * 10_000),
                        "high_minus_low_bps": float((high.mean() - low.mean()) * 10_000),
                    }
                )
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.DataFrame(yearly_rows),
    )


def evaluate_primary_test(
    features: pd.DataFrame,
    signal_column: str,
    outcome_columns: Iterable[str],
    development_start: str,
    development_end: str,
    evaluation_periods: dict[str, tuple[str, str]],
    quantile_count: int = 10,
) -> dict[str, object]:
    """冻结分位后输出主检验所需的 IC、单调性、极端组差与年度一致性。"""

    outcomes = list(outcome_columns)
    _require_columns(features, ["trade_time", signal_column, *outcomes], "主检验特征")
    data = features.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    development_mask = data["trade_time"].between(
        pd.Timestamp(development_start),
        pd.Timestamp(development_end) + pd.Timedelta(days=1),
        inclusive="left",
    )
    edges = fit_quantile_edges(data.loc[development_mask, signal_column], quantile_count)
    data["signal_bucket"] = pd.cut(
        data[signal_column], bins=edges, labels=range(1, quantile_count + 1), include_lowest=True
    ).astype("Int64")
    payload: dict[str, object] = {
        "signal": signal_column,
        "quantile_edges": [None if not np.isfinite(value) else float(value) for value in edges],
        "periods": {},
    }
    period_payload: dict[str, object] = {}
    for period_name, (start, end) in evaluation_periods.items():
        period = data.loc[
            data["trade_time"].between(
                pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), inclusive="left"
            )
        ]
        outcome_payload: dict[str, object] = {}
        for outcome in outcomes:
            valid = period.dropna(subset=[signal_column, "signal_bucket", outcome]).copy()
            if valid.empty:
                outcome_payload[outcome] = {"status": "NO_VALID_OBSERVATIONS"}
                continue
            means = valid.groupby("signal_bucket", observed=True)[outcome].mean().sort_index()
            low = valid.loc[valid["signal_bucket"] == 1, outcome]
            high = valid.loc[valid["signal_bucket"] == quantile_count, outcome]
            yearly_spreads: list[dict[str, object]] = []
            valid["year"] = valid["trade_time"].dt.year
            for year, year_data in valid.groupby("year"):
                yearly_low = year_data.loc[year_data["signal_bucket"] == 1, outcome]
                yearly_high = year_data.loc[year_data["signal_bucket"] == quantile_count, outcome]
                if yearly_low.empty or yearly_high.empty:
                    continue
                yearly_spreads.append(
                    {
                        "year": int(year),
                        "p10_minus_p1_bps": float(
                            (yearly_high.mean() - yearly_low.mean()) * 10_000
                        ),
                    }
                )
            positive_years = sum(row["p10_minus_p1_bps"] > 0 for row in yearly_spreads)
            outcome_payload[outcome] = {
                "status": "PASS_METRICS_COMPUTED",
                "observation_count": int(len(valid)),
                "spearman_ic": float(valid[signal_column].corr(valid[outcome], method="spearman")),
                "bucket_monotonicity_spearman": float(
                    pd.Series(means.index.astype(float)).corr(
                        pd.Series(means.to_numpy(dtype=float)), method="spearman"
                    )
                ),
                "p10_minus_p1_bps": None
                if low.empty or high.empty
                else float((high.mean() - low.mean()) * 10_000),
                "p1_mean_bps": None if low.empty else float(low.mean() * 10_000),
                "p10_mean_bps": None if high.empty else float(high.mean() * 10_000),
                "bucket_mean_bps": {
                    str(int(bucket)): float(value * 10_000)
                    for bucket, value in means.items()
                },
                "year_direction_consistency": None
                if not yearly_spreads
                else float(positive_years / len(yearly_spreads)),
                "yearly_spreads": yearly_spreads,
            }
        period_payload[period_name] = outcome_payload
    payload["periods"] = period_payload
    return payload
