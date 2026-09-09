"""全A股代码哈希留出的十因子训练与预测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURE_COLUMNS = (
    "f01_medium_momentum_120_skip_20_rank",
    "f02_momentum_60_skip_20_rank",
    "f03_momentum_20_rank",
    "f04_reversal_5_rank",
    "f05_low_volatility_20_rank",
    "f06_low_volatility_60_rank",
    "f07_drawdown_position_60_rank",
    "f08_liquidity_shock_20_120_rank",
    "f09_range_compression_20_rank",
    "f10_trend_efficiency_60_rank",
)


@dataclass(frozen=True)
class ModelRules:
    signal_start: pd.Timestamp
    signal_end: pd.Timestamp
    rebalance_step: int
    horizon: int
    minimum_amount: float
    maximum_one_lot: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    random_state: int
    target_clip: tuple[float, float]
    half_life_days: float


def build_features(panel: pd.DataFrame, master: pd.DataFrame, benchmark: pd.DataFrame, rules: ModelRules) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(benchmark["date"]).unique()))
    evaluation = calendar[(calendar >= rules.signal_start) & (calendar <= rules.signal_end)]
    signal_dates = set(evaluation[:: rules.rebalance_step])
    source = panel[
        ["date", "con_code", "raw_high", "raw_low", "raw_close", "total_return_close", "volume", "amount"]
    ].copy()
    source["date"] = pd.to_datetime(source["date"])
    source.sort_values(["con_code", "date"], inplace=True)
    pieces: list[pd.DataFrame] = []
    for _, group in source.groupby("con_code", sort=False):
        group = group.reset_index(drop=True)
        total_close = group["total_return_close"]
        log_return = np.log(total_close / total_close.shift(1))
        amount20 = group["amount"].rolling(20, min_periods=20).mean()
        amount120 = group["amount"].rolling(120, min_periods=120).mean()
        daily_range = (group["raw_high"] - group["raw_low"]) / group["raw_close"]
        calculated = pd.DataFrame(
            {
                "raw_medium": np.log(total_close.shift(20) / total_close.shift(120)),
                "raw_momentum60": np.log(total_close.shift(20) / total_close.shift(60)),
                "raw_momentum20": np.log(total_close / total_close.shift(20)),
                "raw_reversal5": -np.log(total_close / total_close.shift(5)),
                "raw_vol20": log_return.rolling(20, min_periods=20).std(),
                "raw_vol60": log_return.rolling(60, min_periods=60).std(),
                "raw_drawdown60": total_close / total_close.rolling(60, min_periods=60).max() - 1.0,
                "average_amount20": amount20,
                "raw_liquidity_shock": np.log(amount20 / amount120),
                "raw_range_compression": -daily_range.rolling(20, min_periods=20).mean(),
                "raw_trend_efficiency": np.log(total_close / total_close.shift(60)).abs() / log_return.abs().rolling(60, min_periods=60).sum(),
            }
        )
        selected = group["date"].isin(signal_dates)
        if selected.any():
            pieces.append(pd.concat([group.loc[selected].reset_index(drop=True), calculated.loc[selected].reset_index(drop=True)], axis=1))
    if not pieces:
        raise ValueError("没有形成任何信号日特征")
    features = pd.concat(pieces, ignore_index=True)
    meta = master[["ts_code", "list_date", "delist_date"]].copy()
    meta["list_date"] = pd.to_datetime(meta["list_date"], errors="coerce")
    meta["delist_date"] = pd.to_datetime(meta["delist_date"], errors="coerce")
    features = features.merge(meta, left_on="con_code", right_on="ts_code", how="left", validate="many_to_one")
    raw_columns = (
        "raw_medium", "raw_momentum60", "raw_momentum20", "raw_reversal5", "raw_vol20",
        "raw_vol60", "raw_drawdown60", "raw_liquidity_shock", "raw_range_compression", "raw_trend_efficiency",
    )
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features["eligible"] = (
        features[list(raw_columns)].notna().all(axis=1)
        & features["average_amount20"].ge(rules.minimum_amount)
        & features["raw_close"].mul(100.0).le(rules.maximum_one_lot)
        & features["volume"].gt(0.0)
        & features["date"].ge(features["list_date"])
        & (features["delist_date"].isna() | features["date"].lt(features["delist_date"]))
    )
    rank_specs = (
        ("raw_medium", True), ("raw_momentum60", True), ("raw_momentum20", True), ("raw_reversal5", True),
        ("raw_vol20", False), ("raw_vol60", False), ("raw_drawdown60", True), ("raw_liquidity_shock", True),
        ("raw_range_compression", True), ("raw_trend_efficiency", True),
    )
    for output, (raw_name, ascending) in zip(FEATURE_COLUMNS, rank_specs, strict=True):
        features[output] = features.loc[features["eligible"]].groupby("date")[raw_name].rank(pct=True, ascending=ascending)
    features["signal_output"] = np.where(features["eligible"], "SIGNAL_READY", "NO_VIEW")
    return features.sort_values(["date", "con_code"]).reset_index(drop=True), calendar


def build_outcomes(features: pd.DataFrame, panel: pd.DataFrame, benchmark: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    positions = {date: index for index, date in enumerate(calendar)}
    mappings = []
    for date in sorted(pd.to_datetime(features["date"].unique())):
        position = positions.get(pd.Timestamp(date))
        if position is not None and position + horizon + 1 < len(calendar):
            mappings.append({"date": date, "entry_date": calendar[position + 1], "maturity_date": calendar[position + horizon + 1]})
    outcome = features[["date", "con_code"]].merge(pd.DataFrame(mappings), on="date", how="left")
    opens = panel[["date", "con_code", "total_return_open"]].copy()
    opens["date"] = pd.to_datetime(opens["date"])
    outcome = outcome.merge(opens.rename(columns={"date": "entry_date", "total_return_open": "entry_open"}), on=["entry_date", "con_code"], how="left").merge(opens.rename(columns={"date": "maturity_date", "total_return_open": "exit_open"}), on=["maturity_date", "con_code"], how="left")
    bench = benchmark[["date", "close"]].copy()
    bench["date"] = pd.to_datetime(bench["date"])
    outcome = outcome.merge(bench.rename(columns={"date": "entry_date", "close": "bench_entry"}), on="entry_date", how="left").merge(bench.rename(columns={"date": "maturity_date", "close": "bench_exit"}), on="maturity_date", how="left")
    outcome["future_excess_log_return"] = np.log(outcome["exit_open"] / outcome["entry_open"]) - np.log(outcome["bench_exit"] / outcome["bench_entry"])
    outcome.replace([np.inf, -np.inf], np.nan, inplace=True)
    return outcome


def fit_model(features: pd.DataFrame, outcomes: pd.DataFrame, rules: ModelRules) -> tuple[HistGradientBoostingRegressor, dict]:
    data = features.loc[features["signal_output"].eq("SIGNAL_READY")].merge(outcomes[["date", "con_code", "maturity_date", "future_excess_log_return"]], on=["date", "con_code"], how="left", validate="one_to_one").dropna(subset=["future_excess_log_return"])
    data = data.loc[pd.to_datetime(data["maturity_date"]).le(rules.signal_end + pd.Timedelta(days=60))].copy()
    y = data["future_excess_log_return"].clip(*rules.target_clip)
    latest = pd.to_datetime(data["maturity_date"]).max()
    age = (latest - pd.to_datetime(data["maturity_date"])).dt.days.clip(lower=0)
    weights = np.power(0.5, age / rules.half_life_days)
    model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=rules.learning_rate, max_iter=rules.max_iter, max_leaf_nodes=rules.max_leaf_nodes, min_samples_leaf=rules.min_samples_leaf, l2_regularization=rules.l2_regularization, max_bins=rules.max_bins, early_stopping=False, random_state=rules.random_state)
    model.fit(data[list(FEATURE_COLUMNS)].astype(float), y.astype(float), sample_weight=weights)
    return model, {"training_rows": int(len(data)), "signal_dates": int(data["date"].nunique()), "last_maturity": str(pd.Timestamp(data["maturity_date"].max()).date()), "target_mean": float(y.mean()), "target_std": float(y.std())}


def predict_and_select(model: HistGradientBoostingRegressor, features: pd.DataFrame, holdings: int, retention_rank: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = features.loc[features["signal_output"].eq("SIGNAL_READY")].copy()
    ready["predicted_excess_log_return"] = model.predict(ready[list(FEATURE_COLUMNS)].astype(float))
    ready.sort_values(["date", "predicted_excess_log_return", "con_code"], ascending=[True, False, True], inplace=True)
    ready["prediction_rank"] = ready.groupby("date").cumcount() + 1
    previous: list[str] = []
    rows = []
    for date, frame in ready.groupby("date", sort=True):
        ranks = dict(zip(frame["con_code"].astype(str), frame["prediction_rank"], strict=True))
        selected = [code for code in previous if ranks.get(code, retention_rank + 1) <= retention_rank][:holdings]
        selected.extend(code for code in frame["con_code"].astype(str) if code not in selected and len(selected) < holdings)
        day = frame.loc[frame["con_code"].astype(str).isin(selected)].copy()
        order = {code: index + 1 for index, code in enumerate(selected)}
        day["selection_rank"] = day["con_code"].astype(str).map(order)
        rows.append(day)
        previous = selected
    return ready.reset_index(drop=True), pd.concat(rows, ignore_index=True)
