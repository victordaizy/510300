"""构建预注册的510300日线策略因子与可执行收益标签。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutableTargetCosts:
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0003
    minimum_commission_cny: float = 5.0
    stamp_duty_rate: float = 0.0
    slippage_bps: float = 5.0
    lot_size: int = 100


def rolling_percentile(
    series: pd.Series,
    window: int,
    minimum_history: int,
) -> pd.Series:
    """仅使用当前及历史观测计算滚动百分位。"""

    def percentile_last(values: pd.Series) -> float:
        clean = values.dropna()
        if clean.empty:
            return np.nan
        return float(clean.rank(method="average", pct=True).iloc[-1])

    return series.rolling(window, min_periods=minimum_history).apply(
        percentile_last,
        raw=False,
    )


def _daily_total_return(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.Series:
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    cash = events.groupby("ex_date")["cash_dividend_per_share"].sum()
    dividend = prices["date"].map(cash).fillna(0.0)
    return (prices["close"] + dividend) / prices["close"].shift(1) - 1.0


def build_registered_daily_factors(
    etf_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    official_pe: pd.DataFrame,
    secondary_valuation: pd.DataFrame,
    dividends: pd.DataFrame,
    if_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建因子；每一行只使用该日及更早的数据。"""

    etf = etf_daily.copy()
    index = index_daily.copy()
    pe = official_pe.copy()
    valuation = secondary_valuation.copy()
    for frame in (etf, index, pe, valuation):
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    etf_return = _daily_total_return(etf, dividends)
    etf_features = etf[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    etf_features.rename(
        columns={column: f"etf_{column}" for column in ["open", "high", "low", "close", "volume", "amount"]},
        inplace=True,
    )
    etf_features["factor_etf_total_return_5d"] = (
        (1.0 + etf_return).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
    )
    etf_features["factor_etf_total_return_20d"] = (
        (1.0 + etf_return).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    )
    log_volume = np.log1p(etf["volume"].astype(float))
    volume_mean = log_volume.rolling(20, min_periods=20).mean()
    volume_std = log_volume.rolling(20, min_periods=20).std(ddof=1)
    volume_z = (log_volume - volume_mean) / volume_std.replace(0.0, np.nan)
    etf_features["factor_etf_price_volume_confirmation_20d"] = (
        etf_features["factor_etf_total_return_20d"] * volume_z
    )
    daily_vwap = etf["amount"].astype(float) / etf["volume"].astype(float).replace(0.0, np.nan)
    etf_features["factor_etf_close_over_daily_vwap"] = etf["close"] / daily_vwap - 1.0

    index_features = index[["date", "open", "high", "low", "close", "volume"]].copy()
    index_features.rename(
        columns={column: f"index_{column}" for column in ["open", "high", "low", "close", "volume"]},
        inplace=True,
    )
    index_close = index["close"].astype(float)
    index_log_return = np.log(index_close / index_close.shift(1))
    index_features["factor_index_return_20d"] = index_close / index_close.shift(20) - 1.0
    index_features["factor_index_ma20_over_ma60"] = (
        index_close.rolling(20, min_periods=20).mean()
        / index_close.rolling(60, min_periods=60).mean()
        - 1.0
    )
    high_60 = index["high"].rolling(60, min_periods=60).max()
    low_60 = index["low"].rolling(60, min_periods=60).min()
    index_features["factor_index_range_position_60d"] = (
        (index_close - low_60) / (high_60 - low_60).replace(0.0, np.nan)
    )
    rv_5 = index_log_return.rolling(5, min_periods=5).std(ddof=1) * np.sqrt(242.0)
    rv_20 = index_log_return.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(242.0)
    index_features["factor_index_rv_20"] = rv_20
    index_features["factor_index_vol_term_5_20"] = rv_5 / rv_20.replace(0.0, np.nan)

    pe_daily = pe[["date", "pe_official"]].dropna().drop_duplicates("date", keep="last")
    valuation_daily = valuation[["date", "pb"]].dropna().drop_duplicates("date", keep="last")
    warmup_dates = index[["date"]].copy()
    pe_aligned = pd.merge_asof(warmup_dates, pe_daily, on="date", direction="backward")
    pb_aligned = pd.merge_asof(warmup_dates, valuation_daily, on="date", direction="backward")
    valuation_features = warmup_dates.copy()
    valuation_features["official_pe"] = pe_aligned["pe_official"]
    valuation_features["secondary_pb"] = pb_aligned["pb"]
    valuation_features["factor_official_pe_percentile_5y"] = rolling_percentile(
        valuation_features["official_pe"],
        window=242 * 5,
        minimum_history=252,
    )
    valuation_features["factor_secondary_pb_percentile_5y"] = rolling_percentile(
        valuation_features["secondary_pb"],
        window=242 * 5,
        minimum_history=252,
    )

    output = index_features.merge(valuation_features, on="date", how="left", validate="one_to_one")
    output = output.merge(etf_features, on="date", how="inner", validate="one_to_one")

    if if_daily is not None and not if_daily.empty:
        futures = if_daily.copy()
        futures["date"] = pd.to_datetime(futures["date"])
        futures.sort_values("date", inplace=True)
        futures["context_if_return_5d"] = futures["close"] / futures["close"].shift(5) - 1.0
        futures["context_if_open_interest_change_5d"] = (
            futures["open_interest"] / futures["open_interest"].shift(5) - 1.0
        )
        futures = futures[["date", "context_if_return_5d", "context_if_open_interest_change_5d"]]
        output = output.merge(futures, on="date", how="left", validate="one_to_one")
        output["context_if_minus_index_return_5d"] = (
            output["context_if_return_5d"]
            - output["index_close"] / output["index_close"].shift(5) + 1.0
        )

    output["signal_asof_date"] = output["date"]
    if output["date"].duplicated().any() or (output["signal_asof_date"] != output["date"]).any():
        raise ValueError("注册因子日期重复或信息截止日异常")
    return output.sort_values("date").reset_index(drop=True)


def add_executable_targets(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    horizons: tuple[int, ...],
    costs: ExecutableTargetCosts,
) -> pd.DataFrame:
    """构建下一交易日开盘入场、固定交易日收盘退出的成本后标签。"""

    data = prices.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    dividend_by_date = events.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()

    def commission(notional: float) -> float:
        return max(costs.minimum_commission_cny, notional * costs.commission_rate)

    for horizon in horizons:
        target_returns: list[float] = []
        gross_returns: list[float] = []
        maes: list[float] = []
        mfes: list[float] = []
        end_dates: list[pd.Timestamp | pd.NaT] = []
        for index_value in range(len(data)):
            entry_index = index_value + 1
            exit_index = index_value + horizon
            if exit_index >= len(data):
                target_returns.append(np.nan)
                gross_returns.append(np.nan)
                maes.append(np.nan)
                mfes.append(np.nan)
                end_dates.append(pd.NaT)
                continue
            entry = float(data.loc[entry_index, "open"])
            exit_price = float(data.loc[exit_index, "close"])
            buy_price = entry * (1.0 + costs.slippage_bps / 10_000.0)
            sell_price = exit_price * (1.0 - costs.slippage_bps / 10_000.0)
            maximum_lots = int(costs.initial_cash // (buy_price * costs.lot_size))
            while maximum_lots > 0:
                shares = maximum_lots * costs.lot_size
                buy_notional = shares * buy_price
                if buy_notional + commission(buy_notional) <= costs.initial_cash + 1e-9:
                    break
                maximum_lots -= 1
            if maximum_lots <= 0:
                raise ValueError("初始资金不足以买入一手510300")
            shares = maximum_lots * costs.lot_size
            buy_notional = shares * buy_price
            cash = costs.initial_cash - buy_notional - commission(buy_notional)
            # 入场日除息不享有分红，从下一个交易日起计算。
            eligible_dates = data.loc[entry_index + 1 : exit_index, "date"]
            dividend_cash = shares * sum(float(dividend_by_date.get(date, 0.0)) for date in eligible_dates)
            sell_notional = shares * sell_price
            ending_cash = (
                cash
                + dividend_cash
                + sell_notional
                - commission(sell_notional)
                - sell_notional * costs.stamp_duty_rate
            )
            target_returns.append(float(ending_cash / costs.initial_cash - 1.0))
            gross_returns.append(float((exit_price + dividend_cash / shares) / entry - 1.0))
            path = data.loc[entry_index:exit_index]
            maes.append(float(path["low"].min() / entry - 1.0))
            mfes.append(float(path["high"].max() / entry - 1.0))
            end_dates.append(pd.Timestamp(data.loc[exit_index, "date"]))
        data[f"exec_total_return_{horizon}d_net"] = target_returns
        data[f"exec_total_return_{horizon}d_gross"] = gross_returns
        data[f"exec_mae_{horizon}d"] = maes
        data[f"exec_mfe_{horizon}d"] = mfes
        data[f"label_end_date_{horizon}d"] = end_dates
    return data


def build_hysteresis_target(
    features: pd.DataFrame,
    factor_column: str,
    expected_sign: str,
    threshold_mode: str,
    percentile_window: int,
    minimum_history: int,
    entry_percentile: float,
    exit_percentile: float,
    initial_position: float = 0.0,
) -> pd.DataFrame:
    """按预注册30/70滞回规则构造单因子目标仓位。"""

    data = features[["date", factor_column]].copy().sort_values("date").reset_index(drop=True)
    if threshold_mode == "rolling_percentile":
        state_value = rolling_percentile(
            data[factor_column],
            window=percentile_window,
            minimum_history=minimum_history,
        )
    elif threshold_mode == "unit_interval":
        state_value = data[factor_column]
    else:
        raise ValueError(f"未知阈值模式：{threshold_mode}")
    position = float(initial_position)
    targets: list[float] = []
    for value in state_value:
        if pd.notna(value):
            if expected_sign == "positive":
                if value >= entry_percentile:
                    position = 1.0
                elif value <= exit_percentile:
                    position = 0.0
            elif expected_sign == "negative":
                if value <= exit_percentile:
                    position = 1.0
                elif value >= entry_percentile:
                    position = 0.0
            else:
                raise ValueError(f"未知预期方向：{expected_sign}")
        targets.append(position)
    return pd.DataFrame(
        {
            "date": data["date"],
            "target_position": targets,
            "factor_state_value": state_value,
        }
    )
