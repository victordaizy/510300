"""X1成分股共同左尾信号、未来坏状态目标与严格走步预测。"""

from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class X1Rules:
    """冻结的X1信号、目标、概率模型和否证门槛。"""

    data_cutoff: pd.Timestamp
    component_lookback: int
    component_tail_quantile: float
    component_minimum_history: int
    minimum_signal_weight_coverage: float
    risk_minimum_history: int
    top_risk_percentile: float
    industry_shock_percentile: float
    cash_annual_rate: float
    trading_days_per_year: int
    primary_horizon: int
    primary_bad_threshold: float
    primary_tail_threshold: float
    robust_horizon: int
    robust_bad_threshold: float
    robust_tail_threshold: float
    minimum_probability_training: int
    probability_clip: tuple[float, float]
    logistic_c: float
    r6_features: tuple[str, ...]
    augmented_features: tuple[str, ...]
    pseudo_oos_start: pd.Timestamp
    bootstrap_repetitions: int
    bootstrap_block_length: int
    random_seed: int
    minimum_bad20_lift: float
    bootstrap_lower_bound: float
    minimum_brier_skill: float
    minimum_delay_retention: float
    split_lift_lower_bound: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "X1Rules":
        protocol = contract["protocol"]
        signal = contract["signal"]
        targets = contract["targets"]
        probability = contract["probability"]
        evaluation = contract["evaluation"]
        primary = targets["primary"]
        robust = targets["robustness"]
        clip = probability["probability_clip"]
        return cls(
            data_cutoff=pd.Timestamp(protocol["historical_contamination_cutoff"]),
            component_lookback=int(signal["component_return_lookback_trading_days"]),
            component_tail_quantile=float(signal["component_tail_quantile"]),
            component_minimum_history=int(
                signal["component_tail_minimum_prior_observations"]
            ),
            minimum_signal_weight_coverage=float(
                signal["minimum_signal_weight_coverage"]
            ),
            risk_minimum_history=int(
                signal["risk_standardization_minimum_prior_signal_days"]
            ),
            top_risk_percentile=float(signal["top_risk_group_lower_percentile"]),
            industry_shock_percentile=float(
                signal["industry_shock_prior_percentile"]
            ),
            cash_annual_rate=float(targets["cash_annual_rate"]),
            trading_days_per_year=int(targets["trading_days_per_year"]),
            primary_horizon=int(primary["horizon_trading_days"]),
            primary_bad_threshold=float(primary["bad_threshold"]),
            primary_tail_threshold=float(primary["tail_path_threshold"]),
            robust_horizon=int(robust["horizon_trading_days"]),
            robust_bad_threshold=float(robust["bad_threshold"]),
            robust_tail_threshold=float(robust["tail_path_threshold"]),
            minimum_probability_training=int(
                probability["minimum_prior_mature_observations"]
            ),
            probability_clip=(float(clip[0]), float(clip[1])),
            logistic_c=float(probability["r6_baseline_c"]),
            r6_features=tuple(probability["r6_baseline_features"]),
            augmented_features=tuple(probability["r6_plus_x1_features"]),
            pseudo_oos_start=pd.Timestamp(
                evaluation["historical_pseudo_oos_start"]
            ),
            bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
            bootstrap_block_length=int(
                evaluation["bootstrap_block_length_trading_days"]
            ),
            random_seed=int(evaluation["random_seed"]),
            minimum_bad20_lift=float(evaluation["bad20_lift_minimum"]),
            bootstrap_lower_bound=float(
                evaluation["lift_bootstrap_lower_bound_strictly_above"]
            ),
            minimum_brier_skill=float(
                evaluation["brier_skill_strictly_above"]
            ),
            minimum_delay_retention=float(
                evaluation["delayed_excess_lift_retention_minimum"]
            ),
            split_lift_lower_bound=float(
                evaluation["split_lift_strictly_above"]
            ),
        )


def _require_columns(
    frame: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name}缺少字段：{missing}")


def _prior_empirical_percentile(
    values: Sequence[float], minimum_history: int
) -> np.ndarray:
    """当前值相对严格过去值的扩展经验分位。"""

    history: list[float] = []
    result = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        number = float(value) if pd.notna(value) else np.nan
        if np.isfinite(number):
            if len(history) >= minimum_history:
                result[index] = bisect_right(history, number) / len(history)
            insort(history, number)
    return result


def _active_industry_rows(
    cross: pd.DataFrame, industry_intervals: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(
        industry_intervals,
        {"con_code", "industry_l1", "in_date", "out_date"},
        "申万行业区间",
    )
    intervals = industry_intervals[
        ["con_code", "industry_l1", "in_date", "out_date"]
    ].copy()
    intervals["in_date"] = pd.to_datetime(intervals["in_date"], errors="coerce")
    intervals["out_date"] = pd.to_datetime(intervals["out_date"], errors="coerce")
    joined = cross.merge(intervals, on="con_code", how="left")
    active = joined.loc[
        joined["in_date"].le(joined["date"])
        & (joined["out_date"].isna() | joined["out_date"].gt(joined["date"]))
    ].copy()
    if active[["date", "con_code"]].duplicated().any():
        raise ValueError("同一证券日期存在重复有效申万一级行业")
    return active


def build_x1_common_tail_signal(
    constituent_daily: pd.DataFrame,
    component_history: pd.DataFrame,
    weights: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: X1Rules,
) -> pd.DataFrame:
    """只用当前及过去成分信息构造X1，不读取未来ETF收益。"""

    _require_columns(
        constituent_daily,
        {"date", "con_code", "is_index_member"},
        "点时成分成员日线",
    )
    _require_columns(
        component_history,
        {"date", "con_code", "total_return_close"},
        "完整个股价格历史",
    )
    _require_columns(weights, {"trade_date", "con_code", "weight"}, "官方权重")
    members = constituent_daily[["date", "con_code", "is_index_member"]].copy()
    members["date"] = pd.to_datetime(members["date"], errors="coerce")
    members = members.loc[
        members["date"].le(rules.data_cutoff)
        & members["is_index_member"].astype(bool)
    ].copy()
    if members[["date", "con_code"]].duplicated().any():
        raise ValueError("点时成员日线存在重复证券日期")
    counts = members.groupby("date")["con_code"].nunique()
    if counts.empty or not counts.eq(300).all():
        raise ValueError("X1要求每个信号日恰有300只点时成员")

    dates = pd.DatetimeIndex(sorted(members["date"].unique()))
    date_position = pd.Series(np.arange(len(dates)), index=dates)
    universe = sorted(members["con_code"].astype(str).unique())
    history = component_history[["date", "con_code", "total_return_close"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["total_return_close"] = pd.to_numeric(
        history["total_return_close"], errors="coerce"
    )
    history = history.loc[
        history["date"].isin(dates) & history["con_code"].astype(str).isin(universe)
    ].copy()
    if history[["date", "con_code"]].duplicated().any():
        raise ValueError("完整个股价格历史存在重复证券日期")
    full_index = pd.MultiIndex.from_product(
        [universe, dates], names=["con_code", "date"]
    )
    history = (
        history.set_index(["con_code", "date"])
        .reindex(full_index)
        .sort_index()
        .reset_index()
    )
    history["total_return_close"] = history.groupby("con_code", sort=False)[
        "total_return_close"
    ].ffill()
    history["market_position"] = history["date"].map(date_position).astype(int)
    history.sort_values(["con_code", "date"], inplace=True)
    grouped = history.groupby("con_code", sort=False)
    lag_close = grouped["total_return_close"].shift(rules.component_lookback)
    lag_position = grouped["market_position"].shift(rules.component_lookback)
    consecutive = history["market_position"].sub(lag_position).eq(
        rules.component_lookback
    )
    history["component_return_5d"] = (
        history["total_return_close"].div(lag_close).sub(1.0).where(consecutive)
    )
    history["component_tail_threshold"] = grouped[
        "component_return_5d"
    ].transform(
        lambda values: values.shift(1)
        .rolling(
            rules.component_minimum_history,
            min_periods=rules.component_minimum_history,
        )
        .quantile(rules.component_tail_quantile)
    )
    history["component_in_tail"] = (
        history["component_return_5d"].le(history["component_tail_threshold"])
        & history["component_return_5d"].notna()
        & history["component_tail_threshold"].notna()
    )
    daily = members[["date", "con_code"]].merge(
        history[
            [
                "date",
                "con_code",
                "component_return_5d",
                "component_tail_threshold",
                "component_in_tail",
            ]
        ],
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )

    point_weights = weights[["trade_date", "con_code", "weight"]].copy()
    point_weights["trade_date"] = pd.to_datetime(
        point_weights["trade_date"], errors="coerce"
    )
    point_weights["snapshot_weight"] = pd.to_numeric(
        point_weights["weight"], errors="coerce"
    ).div(100.0)
    if point_weights[["trade_date", "con_code"]].duplicated().any():
        raise ValueError("官方权重存在重复证券快照")
    weight_dates = pd.DataFrame(
        {"weight_date": sorted(point_weights["trade_date"].dropna().unique())}
    )
    date_map = pd.merge_asof(
        pd.DataFrame({"date": dates}),
        weight_dates,
        left_on="date",
        right_on="weight_date",
        direction="backward",
    )
    if date_map["weight_date"].isna().any():
        raise ValueError("存在没有事前官方权重快照的信号日")
    point_weights.rename(columns={"trade_date": "weight_date"}, inplace=True)
    daily = daily.merge(date_map, on="date", how="left", validate="many_to_one")
    daily = daily.merge(
        point_weights[["weight_date", "con_code", "snapshot_weight"]],
        on=["weight_date", "con_code"],
        how="left",
        validate="many_to_one",
    )
    daily["tail_eligible"] = (
        daily["component_return_5d"].notna()
        & daily["component_tail_threshold"].notna()
        & daily["snapshot_weight"].notna()
        & daily["snapshot_weight"].gt(0)
    )

    rows: list[dict[str, Any]] = []
    for date, group in daily.groupby("date", sort=True):
        matched_weight = float(group["snapshot_weight"].sum(min_count=1))
        eligible = group["tail_eligible"]
        eligible_weight = float(group.loc[eligible, "snapshot_weight"].sum())
        coverage = eligible_weight / matched_weight if matched_weight > 0 else np.nan
        raw = (
            float(
                group.loc[eligible & group["component_in_tail"], "snapshot_weight"].sum()
                / eligible_weight
            )
            if eligible_weight > 0
            and coverage >= rules.minimum_signal_weight_coverage
            else np.nan
        )
        rows.append(
            {
                "date": pd.Timestamp(date),
                "weight_date": pd.Timestamp(group["weight_date"].iloc[0]),
                "member_count": int(group["con_code"].nunique()),
                "weight_matched_member_count": int(
                    group["snapshot_weight"].notna().sum()
                ),
                "tail_eligible_member_count": int(eligible.sum()),
                "matched_weight": matched_weight,
                "tail_eligible_weight": eligible_weight,
                "tail_signal_weight_coverage": coverage,
                "x1_raw_common_tail_weight": raw,
            }
        )
    signal = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    signal["x1_risk_percentile"] = _prior_empirical_percentile(
        signal["x1_raw_common_tail_weight"].to_numpy(float),
        rules.risk_minimum_history,
    )

    industry_cross = _active_industry_rows(
        daily.loc[daily["tail_eligible"]].copy(), industry_intervals
    )
    industry_daily = (
        industry_cross.groupby(["industry_l1", "date"], sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "industry_tail_share": float(
                        group.loc[group["component_in_tail"], "snapshot_weight"].sum()
                        / group["snapshot_weight"].sum()
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    industry_daily.sort_values(["industry_l1", "date"], inplace=True)
    industry_daily["industry_tail_percentile"] = industry_daily.groupby(
        "industry_l1", sort=False
    )["industry_tail_share"].transform(
        lambda values: _prior_empirical_percentile(
            values.to_numpy(float), rules.risk_minimum_history
        )
    )
    industry_daily["industry_tail_shock"] = industry_daily[
        "industry_tail_percentile"
    ].ge(rules.industry_shock_percentile)
    industry_ready = industry_daily.loc[
        industry_daily["industry_tail_percentile"].notna()
    ]
    breadth = industry_ready.groupby("date").agg(
        industry_tail_breadth=("industry_tail_shock", "mean"),
        industry_count_with_history=("industry_l1", "nunique"),
    )
    signal = signal.merge(breadth, left_on="date", right_index=True, how="left")
    signal["signal_output"] = np.select(
        [
            signal["tail_signal_weight_coverage"].lt(
                rules.minimum_signal_weight_coverage
            ),
            signal["x1_raw_common_tail_weight"].isna(),
            signal["x1_risk_percentile"].isna(),
        ],
        ["NO_VIEW", "NO_VIEW", "WARMUP"],
        default="SIGNAL_READY",
    )
    signal["failure_category"] = np.select(
        [
            signal["tail_signal_weight_coverage"].lt(
                rules.minimum_signal_weight_coverage
            ),
            signal["x1_raw_common_tail_weight"].isna(),
            signal["x1_risk_percentile"].isna(),
        ],
        [
            "TAIL_HISTORY_WEIGHT_COVERAGE_BELOW_GATE",
            "RAW_X1_UNAVAILABLE",
            "RISK_STANDARDIZATION_WARMUP",
        ],
        default="PASS",
    )
    signal["research_output"] = "SIGNAL_AUDIT_ONLY"
    return signal


def _dividend_map(dividends: pd.DataFrame) -> pd.Series:
    _require_columns(
        dividends,
        {"symbol", "ex_date", "cash_dividend_per_share"},
        "510300现金分红",
    )
    cash = dividends.loc[dividends["symbol"].astype(str).eq("510300.SH")].copy()
    cash["ex_date"] = pd.to_datetime(cash["ex_date"], errors="coerce")
    cash["cash_dividend_per_share"] = pd.to_numeric(
        cash["cash_dividend_per_share"], errors="coerce"
    )
    if cash[["ex_date", "cash_dividend_per_share"]].isna().any(axis=None):
        raise ValueError("510300现金分红存在空日期或空金额")
    if (cash["cash_dividend_per_share"] < 0).any():
        raise ValueError("510300现金分红存在负金额")
    return cash.groupby("ex_date")["cash_dividend_per_share"].sum()


def _single_forward_path(
    market: pd.DataFrame,
    dividend_by_date: pd.Series,
    signal_position: int,
    horizon: int,
    delay: int,
    rules: X1Rules,
) -> dict[str, Any] | None:
    entry_position = signal_position + 1 + delay
    maturity_position = entry_position + horizon - 1
    if maturity_position >= len(market):
        return None
    entry = market.iloc[entry_position]
    entry_open = float(entry["open"])
    wealth = float(entry["close"]) / entry_open
    first_cash = (1.0 + rules.cash_annual_rate) ** (
        1.0 / rules.trading_days_per_year
    )
    minimum_path = float(entry["low"]) / entry_open - first_cash
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
        horizon / rules.trading_days_per_year
    )
    return {
        "entry_date": pd.Timestamp(entry["date"]),
        "maturity_date": pd.Timestamp(market.iloc[maturity_position]["date"]),
        "relative_return": float(wealth - cash_wealth),
        "minimum_relative_path": float(minimum_path),
    }


def build_return_tail_outcomes(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp],
    rules: X1Rules,
) -> pd.DataFrame:
    """构造t+1、延迟t+2和60日稳健性未来结果。"""

    _require_columns(etf_daily, {"date", "open", "low", "close"}, "510300日线")
    market = etf_daily[["date", "open", "low", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    for column in ("open", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.loc[market["date"].le(rules.data_cutoff)].sort_values("date")
    market.reset_index(drop=True, inplace=True)
    if market.isna().any(axis=None) or (market[["open", "low", "close"]] <= 0).any(
        axis=None
    ):
        raise ValueError("510300日线存在空值或非正价格")
    if market["date"].duplicated().any():
        raise ValueError("510300日线存在重复日期")
    positions = pd.Series(np.arange(len(market)), index=market["date"])
    dividend_by_date = _dividend_map(dividends)
    rows: list[dict[str, Any]] = []
    for value in signal_dates:
        signal_date = pd.Timestamp(value)
        signal_position = positions.get(signal_date)
        if signal_position is None:
            continue
        primary = _single_forward_path(
            market,
            dividend_by_date,
            int(signal_position),
            rules.primary_horizon,
            0,
            rules,
        )
        delayed = _single_forward_path(
            market,
            dividend_by_date,
            int(signal_position),
            rules.primary_horizon,
            1,
            rules,
        )
        robust = _single_forward_path(
            market,
            dividend_by_date,
            int(signal_position),
            rules.robust_horizon,
            0,
            rules,
        )
        if primary is None:
            continue
        row: dict[str, Any] = {
            "signal_date": signal_date,
            "entry_date20": primary["entry_date"],
            "maturity_date20": primary["maturity_date"],
            "x20": primary["relative_return"],
            "min_path20": primary["minimum_relative_path"],
            "bad20": bool(
                primary["relative_return"] <= rules.primary_bad_threshold
            ),
            "tail20": bool(
                primary["minimum_relative_path"] <= rules.primary_tail_threshold
            ),
        }
        if delayed is not None:
            row.update(
                {
                    "entry_date20_delay1": delayed["entry_date"],
                    "maturity_date20_delay1": delayed["maturity_date"],
                    "x20_delay1": delayed["relative_return"],
                    "min_path20_delay1": delayed["minimum_relative_path"],
                    "bad20_delay1": bool(
                        delayed["relative_return"] <= rules.primary_bad_threshold
                    ),
                    "tail20_delay1": bool(
                        delayed["minimum_relative_path"]
                        <= rules.primary_tail_threshold
                    ),
                }
            )
        if robust is not None:
            row.update(
                {
                    "entry_date60": robust["entry_date"],
                    "maturity_date60": robust["maturity_date"],
                    "x60": robust["relative_return"],
                    "min_path60": robust["minimum_relative_path"],
                    "bad60": bool(
                        robust["relative_return"] <= rules.robust_bad_threshold
                    ),
                    "tail60": bool(
                        robust["minimum_relative_path"]
                        <= rules.robust_tail_threshold
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_r6_information_features(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    valuation: pd.DataFrame,
    rules: X1Rules,
) -> pd.DataFrame:
    """构造冻结R6信息集合的慢估值、趋势与实现波动基准。"""

    _require_columns(etf_daily, {"date", "close"}, "510300日线")
    _require_columns(valuation, {"date", "pe_ttm", "pb"}, "沪深300估值")
    market = etf_daily[["date", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.loc[market["date"].le(rules.data_cutoff)].sort_values("date")
    dividend_by_date = _dividend_map(dividends)
    market["dividend"] = market["date"].map(dividend_by_date).fillna(0.0)
    previous_close = market["close"].shift(1)
    total_return = (market["close"] + market["dividend"]).div(previous_close)
    total_return.iloc[0] = 1.0
    market["total_return_index"] = total_return.cumprod()
    market["log_return"] = np.log(total_return.where(total_return > 0))
    market["trend20"] = market["total_return_index"].div(
        market["total_return_index"].shift(20)
    ).sub(1.0)
    market["trend120"] = market["total_return_index"].div(
        market["total_return_index"].shift(120)
    ).sub(1.0)
    market["rv20"] = market["log_return"].rolling(20, min_periods=20).std() * np.sqrt(
        rules.trading_days_per_year
    )
    market["rv60"] = market["log_return"].rolling(60, min_periods=60).std() * np.sqrt(
        rules.trading_days_per_year
    )
    point_valuation = valuation[["date", "pe_ttm", "pb"]].copy()
    point_valuation["date"] = pd.to_datetime(point_valuation["date"], errors="coerce")
    point_valuation["pe_ttm"] = pd.to_numeric(
        point_valuation["pe_ttm"], errors="coerce"
    )
    point_valuation["pb"] = pd.to_numeric(point_valuation["pb"], errors="coerce")
    point_valuation["earnings_yield"] = 1.0 / point_valuation["pe_ttm"].where(
        point_valuation["pe_ttm"] > 0
    )
    point_valuation["book_yield"] = 1.0 / point_valuation["pb"].where(
        point_valuation["pb"] > 0
    )
    columns = ["date", "trend20", "trend120", "rv20", "rv60"]
    return market[columns].merge(
        point_valuation[["date", "earnings_yield", "book_yield"]],
        on="date",
        how="left",
        validate="one_to_one",
    )


def _training_median_frames(
    train: pd.DataFrame, score: pd.DataFrame, features: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_values = train.loc[:, features].copy()
    score_values = score.loc[:, features].copy()
    medians = train_values.median(axis=0, skipna=True)
    missing = sorted(str(name) for name in medians.index[medians.isna()])
    if missing:
        raise ValueError(f"R6基准训练窗口特征全缺失：{missing}")
    train_values = train_values.fillna(medians)
    score_values = score_values.fillna(medians)
    if score_values.isna().any(axis=None):
        raise ValueError("R6基准评分窗口存在无法插补的特征")
    return train_values, score_values


def _logistic_probability(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: Sequence[str],
    target: str,
    rules: X1Rules,
) -> float:
    train_values, score_values = _training_median_frames(train, score, features)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=rules.logistic_c,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=rules.random_seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_values, train[target].astype(int))
    return float(pipeline.predict_proba(score_values)[0, 1])


def walk_forward_x1_probabilities(
    signal: pd.DataFrame,
    outcomes: pd.DataFrame,
    r6_features: pd.DataFrame,
    rules: X1Rules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个信号日只用此前已成熟BAD20标签拟合概率。"""

    features = signal.merge(
        r6_features, on="date", how="left", validate="one_to_one"
    ).merge(
        outcomes,
        left_on="date",
        right_on="signal_date",
        how="left",
        validate="one_to_one",
    )
    analysis = features.loc[
        features["signal_output"].eq("SIGNAL_READY")
        & features["maturity_date20"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    for score in analysis.sort_values("date").itertuples(index=False):
        signal_date = pd.Timestamp(score.date)
        if signal_date < rules.pseudo_oos_start:
            continue
        train = analysis.loc[
            analysis["maturity_date20"].lt(signal_date)
            & analysis["bad20"].notna()
        ].copy()
        if len(train) < rules.minimum_probability_training:
            continue
        if train["bad20"].astype(int).nunique() < 2:
            continue
        score_frame = analysis.loc[analysis["date"].eq(signal_date)].copy()
        y = train["bad20"].astype(int).to_numpy()
        unconditional = float((y.sum() + 1.0) / (len(y) + 2.0))
        isotonic = IsotonicRegression(
            y_min=rules.probability_clip[0],
            y_max=rules.probability_clip[1],
            increasing=True,
            out_of_bounds="clip",
        )
        isotonic.fit(train["x1_risk_percentile"].to_numpy(float), y)
        x1_probability = float(
            isotonic.predict(score_frame["x1_risk_percentile"].to_numpy(float))[0]
        )
        r6_probability = _logistic_probability(
            train, score_frame, rules.r6_features, "bad20", rules
        )
        augmented_probability = _logistic_probability(
            train, score_frame, rules.augmented_features, "bad20", rules
        )
        rows.append(
            {
                "date": signal_date,
                "maturity_date20": pd.Timestamp(score.maturity_date20),
                "training_mature_observations": int(len(train)),
                "training_bad20_events": int(y.sum()),
                "x1_raw_common_tail_weight": float(score.x1_raw_common_tail_weight),
                "x1_risk_percentile": float(score.x1_risk_percentile),
                "unconditional_probability": unconditional,
                "x1_probability": float(
                    np.clip(x1_probability, *rules.probability_clip)
                ),
                "r6_probability": float(
                    np.clip(r6_probability, *rules.probability_clip)
                ),
                "r6_plus_x1_probability": float(
                    np.clip(augmented_probability, *rules.probability_clip)
                ),
                "bad20": bool(score.bad20),
                "tail20": bool(score.tail20),
                "x20": float(score.x20),
                "bad20_delay1": bool(score.bad20_delay1),
                "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
                "research_output": "PREDICTION_AUDIT_ONLY",
            }
        )
    return analysis, pd.DataFrame(rows)


def _rate_and_lift(
    frame: pd.DataFrame, selected: pd.Series, target: str
) -> dict[str, Any]:
    valid = frame[target].notna()
    base = frame.loc[valid, target].astype(float)
    chosen = frame.loc[valid & selected, target].astype(float)
    baseline_rate = float(base.mean()) if len(base) else np.nan
    selected_rate = float(chosen.mean()) if len(chosen) else np.nan
    lift = (
        selected_rate / baseline_rate
        if baseline_rate > 0 and np.isfinite(selected_rate)
        else np.nan
    )
    return {
        "observations": int(len(base)),
        "selected_observations": int(len(chosen)),
        "event_count": int(base.sum()) if len(base) else 0,
        "selected_event_count": int(chosen.sum()) if len(chosen) else 0,
        "baseline_rate": baseline_rate,
        "selected_rate": selected_rate,
        "lift": float(lift) if np.isfinite(lift) else np.nan,
    }


def _moving_block_bootstrap_lift(
    target: np.ndarray,
    selected: np.ndarray,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    n = len(target)
    if n < 4:
        return {"observations": n, "repetitions": 0, "interval_95pct": None}
    length = min(max(1, block_length), n)
    starts = np.arange(0, n - length + 1)
    blocks_needed = int(np.ceil(n / length))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        chosen_starts = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + length) for start in chosen_starts]
        )[:n]
        sample_target = target[indices].astype(float)
        sample_selected = selected[indices].astype(bool)
        baseline_rate = float(sample_target.mean())
        if baseline_rate <= 0 or not sample_selected.any():
            continue
        lift = float(sample_target[sample_selected].mean() / baseline_rate)
        if np.isfinite(lift):
            values.append(lift)
    if not values:
        return {
            "observations": n,
            "repetitions": 0,
            "interval_95pct": None,
        }
    interval = np.quantile(np.asarray(values), [0.025, 0.975])
    return {
        "observations": n,
        "block_length_trading_days": length,
        "repetitions": int(len(values)),
        "median_lift": float(np.median(values)),
        "interval_95pct": [float(interval[0]), float(interval[1])],
    }


def _brier(probability: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(probability - target)))


def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    )


def _top_decile_mask(values: pd.Series) -> pd.Series:
    return values.rank(method="first", pct=True).gt(0.90)


def evaluate_x1_prediction(
    analysis: pd.DataFrame,
    forecasts: pd.DataFrame,
    rules: X1Rules,
) -> dict[str, Any]:
    """按父协议和实现冻结门槛评价X1，不映射仓位。"""

    primary = analysis.loc[
        analysis["date"].ge(rules.pseudo_oos_start)
        & analysis["bad20"].notna()
    ].sort_values("date")
    if primary.empty or forecasts.empty:
        raise ValueError("没有成熟的X1历史伪样本外预测")
    high_risk = primary["x1_risk_percentile"].ge(rules.top_risk_percentile)
    bad20 = _rate_and_lift(primary, high_risk, "bad20")
    tail20 = _rate_and_lift(primary, high_risk, "tail20")
    delayed = _rate_and_lift(primary, high_risk, "bad20_delay1")
    robust = _rate_and_lift(primary, high_risk, "bad60")
    robust_tail = _rate_and_lift(primary, high_risk, "tail60")
    delay_retention = (
        (delayed["lift"] - 1.0) / (bad20["lift"] - 1.0)
        if pd.notna(delayed["lift"])
        and pd.notna(bad20["lift"])
        and bad20["lift"] > 1.0
        else np.nan
    )
    midpoint = len(primary) // 2
    first = primary.iloc[:midpoint]
    second = primary.iloc[midpoint:]
    first_lift = _rate_and_lift(
        first,
        first["x1_risk_percentile"].ge(rules.top_risk_percentile),
        "bad20",
    )
    second_lift = _rate_and_lift(
        second,
        second["x1_risk_percentile"].ge(rules.top_risk_percentile),
        "bad20",
    )
    bootstrap = _moving_block_bootstrap_lift(
        primary["bad20"].astype(int).to_numpy(),
        high_risk.to_numpy(bool),
        rules.bootstrap_block_length,
        rules.bootstrap_repetitions,
        rules.random_seed,
    )

    target = forecasts["bad20"].astype(int).to_numpy(float)
    unconditional = forecasts["unconditional_probability"].to_numpy(float)
    x1_probability = forecasts["x1_probability"].to_numpy(float)
    r6_probability = forecasts["r6_probability"].to_numpy(float)
    augmented_probability = forecasts["r6_plus_x1_probability"].to_numpy(float)
    unconditional_brier = _brier(unconditional, target)
    x1_brier = _brier(x1_probability, target)
    r6_brier = _brier(r6_probability, target)
    augmented_brier = _brier(augmented_probability, target)
    unconditional_log_loss = _log_loss(unconditional, target)
    x1_log_loss = _log_loss(x1_probability, target)
    r6_log_loss = _log_loss(r6_probability, target)
    augmented_log_loss = _log_loss(augmented_probability, target)
    brier_skill = 1.0 - x1_brier / unconditional_brier
    calibration_difference = float(np.sum(target - x1_probability))
    calibration_standard_error = float(
        np.sqrt(np.sum(x1_probability * (1.0 - x1_probability)))
    )
    calibration_pass = bool(
        calibration_standard_error > 0
        and abs(calibration_difference) <= 1.96 * calibration_standard_error
    )

    r6_top = _top_decile_mask(forecasts["r6_probability"])
    augmented_top = _top_decile_mask(forecasts["r6_plus_x1_probability"])
    r6_recognition = _rate_and_lift(forecasts, r6_top, "bad20")
    augmented_recognition = _rate_and_lift(forecasts, augmented_top, "bad20")
    r6_unit_avoidance = float((-forecasts["x20"] * r6_top.astype(float)).mean())
    augmented_unit_avoidance = float(
        (-forecasts["x20"] * augmented_top.astype(float)).mean()
    )
    incremental = {
        "r6_brier": r6_brier,
        "r6_plus_x1_brier": augmented_brier,
        "r6_log_loss": r6_log_loss,
        "r6_plus_x1_log_loss": augmented_log_loss,
        "r6_top_decile_bad20": r6_recognition,
        "r6_plus_x1_top_decile_bad20": augmented_recognition,
        "r6_unit_avoidance_contribution": r6_unit_avoidance,
        "r6_plus_x1_unit_avoidance_contribution": augmented_unit_avoidance,
    }
    bootstrap_lower = (
        bootstrap["interval_95pct"][0]
        if bootstrap["interval_95pct"] is not None
        else np.nan
    )
    gates = {
        "bad20_lift": pd.notna(bad20["lift"])
        and bad20["lift"] >= rules.minimum_bad20_lift,
        "bootstrap_lift_lower_bound": pd.notna(bootstrap_lower)
        and bootstrap_lower > rules.bootstrap_lower_bound,
        "brier_skill": pd.notna(brier_skill)
        and brier_skill > rules.minimum_brier_skill,
        "log_loss": x1_log_loss < unconditional_log_loss,
        "calibration_bias": calibration_pass,
        "one_day_delay_retention": pd.notna(delay_retention)
        and delay_retention >= rules.minimum_delay_retention,
        "first_half_direction": pd.notna(first_lift["lift"])
        and first_lift["lift"] > rules.split_lift_lower_bound,
        "second_half_direction": pd.notna(second_lift["lift"])
        and second_lift["lift"] > rules.split_lift_lower_bound,
        "incremental_brier": augmented_brier < r6_brier,
        "incremental_log_loss": augmented_log_loss < r6_log_loss,
        "incremental_bad20_recognition": (
            pd.notna(augmented_recognition["selected_rate"])
            and pd.notna(r6_recognition["selected_rate"])
            and augmented_recognition["selected_rate"]
            > r6_recognition["selected_rate"]
        ),
        "incremental_unit_avoidance": augmented_unit_avoidance
        > r6_unit_avoidance,
    }
    status = (
        "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        if all(gates.values())
        else "HISTORICAL_REJECTED_FROZEN"
    )
    return {
        "status": status,
        "candidate_id": "X1",
        "primary_oos_signal_days": int(len(primary)),
        "probability_oos_predictions": int(len(forecasts)),
        "first_primary_signal_date": str(primary["date"].min().date()),
        "last_mature_primary_signal_date": str(primary["date"].max().date()),
        "bad20": bad20,
        "tail20": tail20,
        "bad20_delay1": delayed,
        "delay_excess_lift_retention": float(delay_retention)
        if pd.notna(delay_retention)
        else np.nan,
        "bad60": robust,
        "tail60": robust_tail,
        "first_half_bad20": first_lift,
        "second_half_bad20": second_lift,
        "bootstrap_bad20_lift": bootstrap,
        "probability": {
            "unconditional_brier": unconditional_brier,
            "x1_brier": x1_brier,
            "brier_skill": brier_skill,
            "unconditional_log_loss": unconditional_log_loss,
            "x1_log_loss": x1_log_loss,
            "calibration_event_minus_probability_sum": calibration_difference,
            "calibration_standard_error": calibration_standard_error,
            "calibration_95pct_compatible": calibration_pass,
        },
        "incremental_vs_r6": incremental,
        "gates": gates,
        "safety": {
            "forecast_eligible_emitted": False,
            "combination_model_built": False,
            "position_policy_built": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }


__all__ = [
    "X1Rules",
    "build_r6_information_features",
    "build_return_tail_outcomes",
    "build_x1_common_tail_signal",
    "evaluate_x1_prediction",
    "walk_forward_x1_probabilities",
]
