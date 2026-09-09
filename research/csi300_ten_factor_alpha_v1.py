"""沪深300十因子非线性横截面预测研究。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FACTOR_COLUMNS = (
    "f01_medium_momentum_rank",
    "f02_short_momentum_rank",
    "f03_reversal_5d_rank",
    "f04_low_volatility_rank",
    "f05_drawdown_position_rank",
    "f06_liquidity_shock_rank",
    "f07_informed_flow_rank",
    "f08_value_quality_composite",
    "f09_index_trend_120d",
    "f10_breadth_above_ma60",
)


@dataclass(frozen=True)
class TenFactorRules:
    """冻结的因子、目标和模型参数。"""

    feature_start: pd.Timestamp
    pseudo_oos_start: pd.Timestamp
    end_date: pd.Timestamp
    rebalance_step: int
    target_horizon: int
    feature_columns: tuple[str, ...]
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    random_state: int
    target_clip: tuple[float, float]
    half_life_days: float
    minimum_training_rows: int
    minimum_training_signal_dates: int
    minimum_average_amount: float

    @classmethod
    def from_contract(cls, contract: dict) -> "TenFactorRules":
        schedule = contract["schedule"]
        model = contract["model"]
        selection = contract["selection"]
        columns = tuple(contract["factor_budget"]["feature_columns"])
        if columns != FACTOR_COLUMNS:
            raise ValueError("冻结因子列与实现不一致")
        return cls(
            feature_start=pd.Timestamp(schedule["feature_start_date"]),
            pseudo_oos_start=pd.Timestamp(schedule["pseudo_oos_start_date"]),
            end_date=pd.Timestamp(schedule["end_date"]),
            rebalance_step=int(schedule["rebalance_every_trading_days"]),
            target_horizon=int(schedule["target_horizon_trading_days"]),
            feature_columns=columns,
            learning_rate=float(model["learning_rate"]),
            max_iter=int(model["max_iter"]),
            max_leaf_nodes=int(model["max_leaf_nodes"]),
            min_samples_leaf=int(model["min_samples_leaf"]),
            l2_regularization=float(model["l2_regularization"]),
            max_bins=int(model["max_bins"]),
            random_state=int(model["random_state"]),
            target_clip=(float(model["target_clip"][0]), float(model["target_clip"][1])),
            half_life_days=float(model["training_half_life_calendar_days"]),
            minimum_training_rows=int(model["minimum_training_rows"]),
            minimum_training_signal_dates=int(model["minimum_training_signal_dates"]),
            minimum_average_amount=float(selection["minimum_20d_average_amount_cny"]),
        )


def load_research_history(files: list[str | Path]) -> pd.DataFrame:
    """加载形成因子与目标所需的完整证券历史。"""

    columns = [
        "date",
        "con_code",
        "raw_open",
        "raw_close",
        "total_return_open",
        "total_return_close",
        "volume",
        "amount",
    ]
    frames = [pd.read_parquet(path, columns=columns) for path in files]
    if not frames:
        raise ValueError("证券历史文件为空")
    history = pd.concat(frames, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    history.sort_values(["date", "con_code"], inplace=True)
    if history[["date", "con_code"]].duplicated().any():
        raise ValueError("证券历史存在重复证券日期")
    return history


def _wide(
    frame: pd.DataFrame,
    value: str,
    calendar: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    return frame.pivot(index="date", columns="con_code", values=value).reindex(
        index=calendar, columns=symbols
    )


def _signal_dates(calendar: pd.DatetimeIndex, rules: TenFactorRules) -> pd.DatetimeIndex:
    eligible = calendar[(calendar >= rules.feature_start) & (calendar <= rules.end_date)]
    if eligible.empty:
        raise ValueError("因子区间没有交易日")
    return eligible[:: rules.rebalance_step]


def _melt_signal_wide(
    wide: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    value_name: str,
) -> pd.DataFrame:
    frame = wide.reindex(signal_dates).copy()
    frame.index.name = "date"
    return frame.reset_index().melt(
        id_vars="date", var_name="con_code", value_name=value_name
    )


def _merge_point_in_time_financials(
    signals: pd.DataFrame,
    financials: pd.DataFrame,
) -> pd.DataFrame:
    finance = financials.copy()
    finance["announcement_date"] = pd.to_datetime(
        finance["announcement_date"], errors="coerce"
    )
    finance = finance.dropna(subset=["announcement_date", "con_code"])
    finance = finance.sort_values(["con_code", "announcement_date", "report_period"])
    finance = finance.drop_duplicates(["con_code", "announcement_date"], keep="last")
    pieces: list[pd.DataFrame] = []
    for code, left in signals.groupby("con_code", sort=False):
        right = finance.loc[
            finance["con_code"].eq(code), ["announcement_date", "bps", "roe"]
        ].sort_values("announcement_date")
        ordered = left.sort_values("date")
        if right.empty:
            ordered = ordered.copy()
            ordered["announcement_date"] = pd.NaT
            ordered["bps"] = np.nan
            ordered["roe"] = np.nan
        else:
            ordered = pd.merge_asof(
                ordered,
                right,
                left_on="date",
                right_on="announcement_date",
                direction="backward",
                allow_exact_matches=True,
            )
        pieces.append(ordered)
    result = pd.concat(pieces, ignore_index=True)
    if result["announcement_date"].notna().any():
        invalid = result["announcement_date"].gt(result["date"]).fillna(False)
        if invalid.any():
            raise AssertionError("点时财报合并读取了未来公告")
    return result


def build_ten_factor_panel(
    history: pd.DataFrame,
    member_panel: pd.DataFrame,
    moneyflow: pd.DataFrame,
    financials: pd.DataFrame,
    index_daily: pd.DataFrame,
    breadth: pd.DataFrame,
    rules: TenFactorRules,
) -> pd.DataFrame:
    """只用信号日及之前信息形成固定十因子截面。"""

    members = member_panel.copy()
    members["date"] = pd.to_datetime(members["date"])
    members = members.loc[members["date"].le(rules.end_date)].copy()
    calendar = pd.DatetimeIndex(sorted(members["date"].unique()))
    symbols = sorted(history["con_code"].astype(str).unique())
    signal_dates = _signal_dates(calendar, rules)

    close_observed = _wide(history, "total_return_close", calendar, symbols)
    close = close_observed.ffill()
    returns = close.pct_change(fill_method=None)
    amount = _wide(history, "amount", calendar, symbols).fillna(0.0)
    raw_close = _wide(history, "raw_close", calendar, symbols)

    medium_momentum = np.log(close.shift(20).divide(close.shift(120)))
    short_momentum = np.log(close.divide(close.shift(20)))
    reversal_5d = -np.log(close.divide(close.shift(5)))
    volatility_20d = returns.rolling(20, min_periods=15).std()
    drawdown_position = close.divide(close.rolling(60, min_periods=40).max()).subtract(1.0)
    amount_mean20 = amount.rolling(20, min_periods=15).mean()
    amount_mean120 = amount.rolling(120, min_periods=80).mean()
    liquidity_shock = np.log(amount_mean20.divide(amount_mean120).where(amount_mean120.gt(0)))

    flow = moneyflow.copy()
    flow["date"] = pd.to_datetime(flow["date"])
    flow.rename(columns={"ts_code": "con_code"}, inplace=True)
    large = _wide(flow, "large_extra_large_net_amount_10k_cny", calendar, symbols).fillna(0.0)
    small = _wide(flow, "small_net_amount_10k_cny", calendar, symbols).fillna(0.0)
    informed_flow = (
        (large.subtract(small) * 10000.0).rolling(20, min_periods=15).sum()
        .divide(amount.rolling(20, min_periods=15).sum())
        .where(amount.rolling(20, min_periods=15).sum().gt(0))
    )

    raw_frames = {
        "raw_medium_momentum": medium_momentum,
        "raw_short_momentum": short_momentum,
        "raw_reversal_5d": reversal_5d,
        "raw_volatility_20d": volatility_20d,
        "raw_drawdown_position": drawdown_position,
        "raw_liquidity_shock": liquidity_shock,
        "raw_informed_flow": informed_flow,
        "average_amount_20d": amount_mean20,
        "raw_close_signal": raw_close,
    }
    signal = members.loc[
        members["date"].isin(signal_dates) & members["is_index_member"].astype(bool),
        ["date", "con_code", "is_suspended"],
    ].copy()
    for name, wide_frame in raw_frames.items():
        signal = signal.merge(
            _melt_signal_wide(wide_frame, signal_dates, name),
            on=["date", "con_code"],
            how="left",
            validate="one_to_one",
        )

    signal = _merge_point_in_time_financials(signal, financials)
    signal["raw_book_to_price"] = signal["bps"].divide(signal["raw_close_signal"])
    grouped = signal.groupby("date", sort=False)
    signal["f01_medium_momentum_rank"] = grouped["raw_medium_momentum"].rank(pct=True)
    signal["f02_short_momentum_rank"] = grouped["raw_short_momentum"].rank(pct=True)
    signal["f03_reversal_5d_rank"] = grouped["raw_reversal_5d"].rank(pct=True)
    signal["f04_low_volatility_rank"] = grouped["raw_volatility_20d"].rank(
        pct=True, ascending=False
    )
    signal["f05_drawdown_position_rank"] = grouped["raw_drawdown_position"].rank(pct=True)
    signal["f06_liquidity_shock_rank"] = grouped["raw_liquidity_shock"].rank(pct=True)
    signal["f07_informed_flow_rank"] = grouped["raw_informed_flow"].rank(pct=True)
    book_rank = grouped["raw_book_to_price"].rank(pct=True)
    roe_rank = grouped["roe"].rank(pct=True)
    signal["f08_value_quality_composite"] = pd.concat(
        [book_rank, roe_rank], axis=1
    ).mean(axis=1, skipna=True)

    index_state = index_daily[["date", "close"]].copy()
    index_state["date"] = pd.to_datetime(index_state["date"])
    index_state.sort_values("date", inplace=True)
    index_state["f09_index_trend_120d"] = index_state["close"].divide(
        index_state["close"].rolling(120, min_periods=120).mean()
    ).subtract(1.0)
    breadth_state = breadth[["date", "equal_above_ma60_share"]].copy()
    breadth_state["date"] = pd.to_datetime(breadth_state["date"])
    breadth_state.rename(
        columns={"equal_above_ma60_share": "f10_breadth_above_ma60"}, inplace=True
    )
    signal = signal.merge(
        index_state[["date", "f09_index_trend_120d"]],
        on="date",
        how="left",
        validate="many_to_one",
    ).merge(
        breadth_state,
        on="date",
        how="left",
        validate="many_to_one",
    )
    signal.replace([np.inf, -np.inf], np.nan, inplace=True)
    signal["factor_available_count"] = signal[list(rules.feature_columns)].notna().sum(axis=1)
    signal["signal_output"] = np.where(
        signal["factor_available_count"].ge(8)
        & signal["average_amount_20d"].ge(rules.minimum_average_amount)
        & ~signal["is_suspended"].astype(bool),
        "SIGNAL_READY",
        "NO_VIEW",
    )
    signal.sort_values(["date", "con_code"], inplace=True)
    signal.reset_index(drop=True, inplace=True)
    return signal


def build_forward_relative_outcomes(
    factors: pd.DataFrame,
    history: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rules: TenFactorRules,
) -> pd.DataFrame:
    """构造t+1开盘至t+11开盘的个股相对基准收益。"""

    dates = pd.DatetimeIndex(calendar)
    positions = {date: index for index, date in enumerate(dates)}
    mapping_rows = []
    for signal_date in sorted(pd.to_datetime(factors["date"].unique())):
        position = positions.get(pd.Timestamp(signal_date))
        exit_position = None if position is None else position + rules.target_horizon + 1
        if position is None or exit_position >= len(dates):
            continue
        mapping_rows.append(
            {
                "date": pd.Timestamp(signal_date),
                "entry_date": dates[position + 1],
                "maturity_date": dates[exit_position],
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    outcomes = factors[["date", "con_code"]].merge(
        mapping, on="date", how="left", validate="many_to_one"
    )
    opens = history[["date", "con_code", "total_return_open"]].copy()
    opens["date"] = pd.to_datetime(opens["date"])
    entry = opens.rename(
        columns={"date": "entry_date", "total_return_open": "entry_total_return_open"}
    )
    exit_frame = opens.rename(
        columns={"date": "maturity_date", "total_return_open": "exit_total_return_open"}
    )
    outcomes = outcomes.merge(
        entry,
        on=["entry_date", "con_code"],
        how="left",
        validate="many_to_one",
    ).merge(
        exit_frame,
        on=["maturity_date", "con_code"],
        how="left",
        validate="many_to_one",
    )
    benchmark_frame = benchmark[["date", "close"]].copy()
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
    outcomes = outcomes.merge(
        benchmark_frame.rename(columns={"date": "entry_date", "close": "benchmark_entry"}),
        on="entry_date",
        how="left",
        validate="many_to_one",
    ).merge(
        benchmark_frame.rename(columns={"date": "maturity_date", "close": "benchmark_exit"}),
        on="maturity_date",
        how="left",
        validate="many_to_one",
    )
    stock_return = np.log(
        outcomes["exit_total_return_open"].divide(outcomes["entry_total_return_open"])
    )
    benchmark_return = np.log(outcomes["benchmark_exit"].divide(outcomes["benchmark_entry"]))
    outcomes["future_excess_log_return_10d"] = stock_return.subtract(benchmark_return)
    outcomes.replace([np.inf, -np.inf], np.nan, inplace=True)
    return outcomes


def walk_forward_predictions(
    factors: pd.DataFrame,
    outcomes: pd.DataFrame,
    rules: TenFactorRules,
) -> pd.DataFrame:
    """每个信号日仅使用此前已成熟标签拟合固定模型。"""

    data = factors.merge(
        outcomes[
            ["date", "con_code", "maturity_date", "future_excess_log_return_10d"]
        ],
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    data["date"] = pd.to_datetime(data["date"])
    data["maturity_date"] = pd.to_datetime(data["maturity_date"])
    predictions: list[pd.DataFrame] = []
    for signal_date in sorted(data.loc[data["date"].ge(rules.pseudo_oos_start), "date"].unique()):
        signal_date = pd.Timestamp(signal_date)
        training = data.loc[
            data["maturity_date"].le(signal_date)
            & data["future_excess_log_return_10d"].notna()
            & data["signal_output"].eq("SIGNAL_READY")
        ].copy()
        current = data.loc[
            data["date"].eq(signal_date) & data["signal_output"].eq("SIGNAL_READY")
        ].copy()
        if current.empty:
            continue
        if (
            len(training) < rules.minimum_training_rows
            or training["date"].nunique() < rules.minimum_training_signal_dates
        ):
            current["model_output"] = "NO_MODEL"
            current["predicted_excess_log_return_10d"] = np.nan
            current["training_rows"] = int(len(training))
            current["training_signal_dates"] = int(training["date"].nunique())
            predictions.append(current)
            continue
        x_train = training[list(rules.feature_columns)].astype(float)
        y_train = training["future_excess_log_return_10d"].clip(*rules.target_clip).astype(float)
        age_days = (signal_date - training["maturity_date"]).dt.days.clip(lower=0).astype(float)
        sample_weight = np.power(0.5, age_days / rules.half_life_days)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=rules.learning_rate,
            max_iter=rules.max_iter,
            max_leaf_nodes=rules.max_leaf_nodes,
            min_samples_leaf=rules.min_samples_leaf,
            l2_regularization=rules.l2_regularization,
            max_bins=rules.max_bins,
            early_stopping=False,
            random_state=rules.random_state,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        current["predicted_excess_log_return_10d"] = model.predict(
            current[list(rules.feature_columns)].astype(float)
        )
        current["model_output"] = "PREDICTION_AUDIT_ONLY"
        current["training_rows"] = int(len(training))
        current["training_signal_dates"] = int(training["date"].nunique())
        current["latest_training_maturity_date"] = training["maturity_date"].max()
        current.sort_values(
            ["predicted_excess_log_return_10d", "con_code"],
            ascending=[False, True],
            inplace=True,
        )
        current["prediction_rank"] = np.arange(1, len(current) + 1)
        predictions.append(current)
    if not predictions:
        return pd.DataFrame()
    result = pd.concat(predictions, ignore_index=True)
    if result["latest_training_maturity_date"].notna().any():
        invalid = result["latest_training_maturity_date"].gt(result["date"]).fillna(False)
        if invalid.any():
            raise AssertionError("走步模型读取了信号日之后成熟的标签")
    return result

