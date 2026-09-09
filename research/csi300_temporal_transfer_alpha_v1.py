"""只用早期数据训练、冻结后迁移到近期的十因子模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURE_COLUMNS = (
    "f01_medium_momentum_rank",
    "f02_momentum_60d_rank",
    "f03_momentum_20d_rank",
    "f04_reversal_5d_rank",
    "f05_low_volatility_20d_rank",
    "f06_low_volatility_60d_rank",
    "f07_drawdown_position_rank",
    "f08_liquidity_shock_rank",
    "f09_index_trend_120d",
    "f10_breadth_above_ma60",
)


@dataclass(frozen=True)
class TransferRules:
    start: pd.Timestamp
    end: pd.Timestamp
    step: int
    horizon: int
    features: tuple[str, ...]
    minimum_amount: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    random_state: int
    target_clip: tuple[float, float]
    half_life_days: float


def build_price_factor_panel(
    history: pd.DataFrame,
    members: pd.DataFrame,
    index_daily: pd.DataFrame,
    rules: TransferRules,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """形成只依赖当日及过去信息的十因子信号表。"""

    member = members.copy()
    member["date"] = pd.to_datetime(member["date"])
    calendar = pd.DatetimeIndex(
        sorted(member.loc[member["date"].le(rules.end), "date"].unique())
    )
    signal_calendar = calendar[(calendar >= rules.start) & (calendar <= rules.end)][:: rules.step]
    market = history.copy()
    market["date"] = pd.to_datetime(market["date"])
    symbols = sorted(market["con_code"].astype(str).unique())

    def wide(column: str) -> pd.DataFrame:
        return market.pivot(index="date", columns="con_code", values=column).reindex(
            index=calendar, columns=symbols
        )

    close = wide("total_return_close").ffill()
    amount = wide("amount").fillna(0.0)
    returns = close.pct_change(fill_method=None)
    raw = {
        "raw_medium_momentum": np.log(close.shift(20).divide(close.shift(120))),
        "raw_momentum_60d": np.log(close.divide(close.shift(60))),
        "raw_momentum_20d": np.log(close.divide(close.shift(20))),
        "raw_reversal_5d": -np.log(close.divide(close.shift(5))),
        "raw_volatility_20d": returns.rolling(20, min_periods=15).std(),
        "raw_volatility_60d": returns.rolling(60, min_periods=40).std(),
        "raw_drawdown_position": close.divide(close.rolling(60, min_periods=40).max()).subtract(1.0),
        "raw_liquidity_shock": np.log(
            amount.rolling(20, min_periods=15).mean().divide(
                amount.rolling(120, min_periods=80).mean()
            )
        ),
        "average_amount_20d": amount.rolling(20, min_periods=15).mean(),
    }
    signal = member.loc[
        member["date"].isin(signal_calendar) & member["is_index_member"].astype(bool),
        ["date", "con_code", "is_suspended"],
    ].copy()
    for name, frame in raw.items():
        long = frame.reindex(signal_calendar).rename_axis("date").reset_index().melt(
            id_vars="date", var_name="con_code", value_name=name
        )
        signal = signal.merge(long, on=["date", "con_code"], how="left", validate="one_to_one")
    grouped = signal.groupby("date", sort=False)
    signal["f01_medium_momentum_rank"] = grouped["raw_medium_momentum"].rank(pct=True)
    signal["f02_momentum_60d_rank"] = grouped["raw_momentum_60d"].rank(pct=True)
    signal["f03_momentum_20d_rank"] = grouped["raw_momentum_20d"].rank(pct=True)
    signal["f04_reversal_5d_rank"] = grouped["raw_reversal_5d"].rank(pct=True)
    signal["f05_low_volatility_20d_rank"] = grouped["raw_volatility_20d"].rank(pct=True, ascending=False)
    signal["f06_low_volatility_60d_rank"] = grouped["raw_volatility_60d"].rank(pct=True, ascending=False)
    signal["f07_drawdown_position_rank"] = grouped["raw_drawdown_position"].rank(pct=True)
    signal["f08_liquidity_shock_rank"] = grouped["raw_liquidity_shock"].rank(pct=True)

    index = index_daily[["date", "close"]].copy()
    index["date"] = pd.to_datetime(index["date"])
    index.sort_values("date", inplace=True)
    index["f09_index_trend_120d"] = index["close"].divide(
        index["close"].rolling(120, min_periods=120).mean()
    ).subtract(1.0)
    above = close.gt(close.rolling(60, min_periods=60).mean())
    membership = member.pivot(index="date", columns="con_code", values="is_index_member").reindex(
        index=calendar, columns=symbols
    ).fillna(False).astype(bool)
    breadth = above.where(membership).mean(axis=1).rename("f10_breadth_above_ma60")
    state = index[["date", "f09_index_trend_120d"]].merge(
        breadth.rename_axis("date").reset_index(), on="date", how="left", validate="one_to_one"
    )
    signal = signal.merge(state, on="date", how="left", validate="many_to_one")
    signal.replace([np.inf, -np.inf], np.nan, inplace=True)
    signal["factor_available_count"] = signal[list(rules.features)].notna().sum(axis=1)
    signal["signal_output"] = np.where(
        signal["factor_available_count"].ge(9)
        & signal["average_amount_20d"].ge(rules.minimum_amount)
        & ~signal["is_suspended"].astype(bool),
        "SIGNAL_READY",
        "NO_VIEW",
    )
    return signal.sort_values(["date", "con_code"]).reset_index(drop=True), calendar


def build_outcomes(
    features: pd.DataFrame,
    history: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """构造下一开盘至固定退出开盘的个股相对H00300收益。"""

    positions = {date: index for index, date in enumerate(calendar)}
    mapping = []
    for date in sorted(pd.to_datetime(features["date"].unique())):
        position = positions[pd.Timestamp(date)]
        if position + horizon + 1 < len(calendar):
            mapping.append({"date": date, "entry_date": calendar[position + 1], "maturity_date": calendar[position + horizon + 1]})
    result = features[["date", "con_code"]].merge(pd.DataFrame(mapping), on="date", how="left")
    opens = history[["date", "con_code", "total_return_open"]].copy()
    opens["date"] = pd.to_datetime(opens["date"])
    result = result.merge(
        opens.rename(columns={"date": "entry_date", "total_return_open": "entry_open"}),
        on=["entry_date", "con_code"], how="left", validate="many_to_one"
    ).merge(
        opens.rename(columns={"date": "maturity_date", "total_return_open": "exit_open"}),
        on=["maturity_date", "con_code"], how="left", validate="many_to_one"
    )
    bench = benchmark[["date", "close"]].copy()
    bench["date"] = pd.to_datetime(bench["date"])
    result = result.merge(bench.rename(columns={"date": "entry_date", "close": "bench_entry"}), on="entry_date", how="left").merge(
        bench.rename(columns={"date": "maturity_date", "close": "bench_exit"}), on="maturity_date", how="left"
    )
    result["future_excess_log_return"] = np.log(result["exit_open"].divide(result["entry_open"])) - np.log(result["bench_exit"].divide(result["bench_entry"]))
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def fit_model(features: pd.DataFrame, outcomes: pd.DataFrame, rules: TransferRules) -> tuple[HistGradientBoostingRegressor, dict]:
    data = features.loc[features["signal_output"].eq("SIGNAL_READY")].merge(
        outcomes[["date", "con_code", "maturity_date", "future_excess_log_return"]],
        on=["date", "con_code"], how="left", validate="one_to_one"
    ).dropna(subset=["future_excess_log_return"])
    data = data.loc[pd.to_datetime(data["maturity_date"]).le(rules.end)].copy()
    y = data["future_excess_log_return"].clip(*rules.target_clip)
    age = (rules.end - pd.to_datetime(data["maturity_date"])).dt.days.clip(lower=0)
    weights = np.power(0.5, age / rules.half_life_days)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=rules.learning_rate, max_iter=rules.max_iter,
        max_leaf_nodes=rules.max_leaf_nodes, min_samples_leaf=rules.min_samples_leaf,
        l2_regularization=rules.l2_regularization, max_bins=rules.max_bins,
        early_stopping=False, random_state=rules.random_state,
    )
    model.fit(data[list(rules.features)].astype(float), y.astype(float), sample_weight=weights)
    report = {
        "training_rows": int(len(data)),
        "training_signal_dates": int(data["date"].nunique()),
        "first_training_signal": str(pd.Timestamp(data["date"].min()).date()),
        "last_training_signal": str(pd.Timestamp(data["date"].max()).date()),
        "last_maturity_date": str(pd.Timestamp(data["maturity_date"].max()).date()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
    }
    return model, report


def predict_and_select(
    model: HistGradientBoostingRegressor,
    features: pd.DataFrame,
    rules: TransferRules,
    holdings: int,
    retention_rank: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = features.loc[features["signal_output"].eq("SIGNAL_READY")].copy()
    ready["predicted_excess_log_return"] = model.predict(ready[list(rules.features)].astype(float))
    ready.sort_values(["date", "predicted_excess_log_return", "con_code"], ascending=[True, False, True], inplace=True)
    ready["prediction_rank"] = ready.groupby("date").cumcount() + 1
    selected_rows = []
    previous: list[str] = []
    for date, frame in ready.groupby("date", sort=True):
        rank_map = dict(zip(frame["con_code"].astype(str), frame["prediction_rank"], strict=True))
        retained = [code for code in previous if rank_map.get(code, retention_rank + 1) <= retention_rank]
        ordered = frame["con_code"].astype(str).tolist()
        selected = retained[:holdings]
        selected.extend(code for code in ordered if code not in selected and len(selected) < holdings)
        day = frame.loc[frame["con_code"].astype(str).isin(selected)].copy()
        order = {code: index + 1 for index, code in enumerate(selected)}
        day["selection_rank"] = day["con_code"].astype(str).map(order)
        selected_rows.append(day)
        previous = selected
    return ready.reset_index(drop=True), pd.concat(selected_rows, ignore_index=True)

