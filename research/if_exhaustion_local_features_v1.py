"""从已冻结父代码提取的纯计算函数；没有采集或回测入口。"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
EXPECTED_STATES = ["NORMAL", "FORCED_SELLING_ACTIVE", "FORCED_SELLING_EXHAUSTED"]

def _assign_maturity_bucket(
    days_to_expiry: pd.Series, buckets: list[list[int]]
) -> pd.Series:
    output = pd.Series(pd.NA, index=days_to_expiry.index, dtype="string")
    for lower, upper in buckets:
        label = f"DTE_{int(lower):03d}_{int(upper):03d}"
        output.loc[days_to_expiry.between(int(lower), int(upper))] = label
    return output

def _trailing_total_return(wealth: pd.Series, days: int) -> pd.Series:
    return wealth / wealth.shift(int(days)) - 1.0

def build_if_features(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contract = config["if_data_contract"]
    joined = inputs["joined_contracts"].copy()
    spot = inputs["spot"][["date", "close"]].rename(
        columns={"close": "spot_close"}
    )
    joined = joined.merge(spot, on="date", how="inner", validate="many_to_one")
    lower = int(contract["eligible_days_to_expiry_min_inclusive"])
    upper = int(contract["eligible_days_to_expiry_max_inclusive"])
    eligible = joined.loc[
        joined["calendar_days_to_expiry"].between(lower, upper)
        & joined[contract["selected_price_field"]].gt(0.0)
        & joined["spot_close"].gt(0.0)
        & joined["open_interest"].ge(0.0)
    ].copy()
    eligible["maturity_bucket"] = _assign_maturity_bucket(
        eligible["calendar_days_to_expiry"],
        contract["maturity_buckets_calendar_days"],
    )
    if eligible["maturity_bucket"].isna().any():
        raise ValueError("合格IF合约未落入冻结期限桶")
    price_field = str(contract["selected_price_field"])
    eligible["annualized_basis"] = (
        np.log(eligible[price_field] / eligible["spot_close"])
        * 365.0
        / eligible["calendar_days_to_expiry"]
    )
    eligible["calendar_month"] = eligible["date"].dt.month.astype(int)
    daily_bucket = (
        eligible.groupby(
            ["date", "calendar_month", "maturity_bucket"],
            as_index=False,
            observed=True,
        )["annualized_basis"]
        .median()
        .rename(columns={"annualized_basis": "daily_bucket_median_basis"})
    )
    daily_bucket.sort_values(
        ["calendar_month", "maturity_bucket", "date"],
        kind="mergesort",
        inplace=True,
    )
    minimum_prior = int(contract["seasonal_baseline_minimum_prior_days"])
    daily_bucket["seasonal_maturity_baseline"] = daily_bucket.groupby(
        ["calendar_month", "maturity_bucket"], observed=True
    )["daily_bucket_median_basis"].transform(
        lambda values: values.expanding(min_periods=minimum_prior).median().shift(1)
    )
    daily_bucket["bucket_basis_residual"] = (
        daily_bucket["daily_bucket_median_basis"]
        - daily_bucket["seasonal_maturity_baseline"]
    )
    basis_daily = (
        daily_bucket.groupby("date", as_index=False)
        .agg(
            basis_residual=("bucket_basis_residual", "median"),
            eligible_maturity_buckets=("maturity_bucket", "nunique"),
            baseline_eligible_buckets=("seasonal_maturity_baseline", "count"),
        )
        .sort_values("date", kind="mergesort")
    )
    oi_daily = (
        joined.loc[
            joined["calendar_days_to_expiry"].between(1, upper)
            & joined["open_interest"].ge(0.0)
        ]
        .groupby("date", as_index=False)
        .agg(
            aggregate_open_interest=("open_interest", "sum"),
            aggregate_oi_contract_count=("symbol", "nunique"),
        )
        .sort_values("date", kind="mergesort")
    )
    raw = inputs["market"][["date"]].merge(
        basis_daily, on="date", how="left", validate="one_to_one"
    ).merge(oi_daily, on="date", how="left", validate="one_to_one")
    raw["open_interest_shock"] = np.log(
        raw["aggregate_open_interest"].replace(0.0, np.nan)
    ).diff(3)
    window = int(contract["rolling_history_trading_days"])
    minimum = int(contract["rolling_history_minimum_trading_days"])
    raw["basis_residual_percentile"] = rolling_prior_percentile(
        raw["basis_residual"], window, minimum
    )
    raw["open_interest_shock_percentile"] = rolling_prior_percentile(
        raw["open_interest_shock"], window, minimum
    )
    raw.rename(columns={"date": "feature_source_date"}, inplace=True)
    calendar = inputs["market"]["date"].reset_index(drop=True)
    next_date_map = {
        pd.Timestamp(calendar.iloc[index]): pd.Timestamp(calendar.iloc[index + 1])
        for index in range(len(calendar) - 1)
    }
    raw["signal_date"] = raw["feature_source_date"].map(next_date_map)
    raw = raw.loc[raw["signal_date"].notna()].copy()

    features = inputs["market"].copy()
    if "amount" not in features.columns:
        raise ValueError("510300行情缺少成交额，无法进行价格控制匹配")
    features["etf_past_3d_total_return"] = _trailing_total_return(
        features["total_wealth_index"], 3
    )
    features["etf_past_5d_total_return"] = _trailing_total_return(
        features["total_wealth_index"], 5
    )
    features["etf_realized_volatility_20d"] = (
        features["total_return"].rolling(20, min_periods=20).std(ddof=1)
        * math.sqrt(242.0)
    )
    features["etf_same_day_total_return"] = features["total_return"]
    prior_amount_median = (
        pd.to_numeric(features["amount"], errors="raise")
        .shift(1)
        .rolling(60, min_periods=40)
        .median()
    )
    features["etf_log_amount_vs_trailing_60d_median"] = np.log(
        pd.to_numeric(features["amount"], errors="raise") / prior_amount_median
    )
    matching_columns = config["matching"]["covariates"]
    for column in matching_columns:
        features[f"rank_{column}"] = rolling_prior_percentile(
            features[column], window, minimum
        )
    features = features.merge(
        raw,
        left_on="date",
        right_on="signal_date",
        how="left",
        validate="one_to_one",
    )
    features["if_feature_available_at"] = features["date"] + pd.Timedelta(
        hours=15, minutes=1
    )
    features["data_eligible"] = features[
        [
            "basis_residual_percentile",
            "open_interest_shock_percentile",
            "etf_past_3d_total_return",
            *[f"rank_{column}" for column in matching_columns],
        ]
    ].notna().all(axis=1)
    audit = {
        "contract_rows_joined_to_spot": int(len(joined)),
        "eligible_contract_rows": int(len(eligible)),
        "daily_bucket_rows": int(len(daily_bucket)),
        "raw_feature_rows": int(len(raw)),
        "aligned_market_rows": int(len(features)),
        "first_basis_percentile_signal_date": (
            features.loc[
                features["basis_residual_percentile"].notna(), "date"
            ].min().date().isoformat()
        ),
        "first_oi_percentile_signal_date": (
            features.loc[
                features["open_interest_shock_percentile"].notna(), "date"
            ].min().date().isoformat()
        ),
        "eligible_signal_days": int(features["data_eligible"].sum()),
        "future_return_columns_read_during_feature_construction": 0,
        "availability_lag_trading_days": int(
            contract["signal_data_availability_lag_trading_days"]
        ),
    }
    return eligible.reset_index(drop=True), features, audit

def build_state_machine(
    config: dict[str, Any], features: pd.DataFrame
) -> pd.DataFrame:
    signal = config["if_signal"]
    output = features.copy().sort_values("date", kind="mergesort").reset_index(
        drop=True
    )
    output["entry_condition"] = (
        output["data_eligible"]
        & output["etf_past_3d_total_return"].lt(0.0)
        & output["basis_residual_percentile"].le(
            float(signal["entry_basis_percentile_max"])
        )
        & output["open_interest_shock_percentile"].ge(
            float(signal["entry_open_interest_percentile_min"])
        )
    )
    output["exhaustion_condition_raw"] = (
        output["data_eligible"]
        & output["basis_residual_percentile"].ge(
            float(signal["exhaustion_basis_percentile_min"])
        )
        & output["open_interest_shock_percentile"].le(
            float(signal["exhaustion_open_interest_percentile_max"])
        )
    )
    states: list[str] = []
    start_events: list[bool] = []
    exhaustion_events: list[bool] = []
    active = False
    consecutive_exhaustion = 0
    required = int(signal["exhaustion_consecutive_trading_days"])
    for row in output.itertuples(index=False):
        start_event = False
        exhaustion_event = False
        if active:
            if bool(row.exhaustion_condition_raw):
                consecutive_exhaustion += 1
            else:
                consecutive_exhaustion = 0
            if consecutive_exhaustion >= required:
                state = "FORCED_SELLING_EXHAUSTED"
                exhaustion_event = True
                active = False
                consecutive_exhaustion = 0
            else:
                state = "FORCED_SELLING_ACTIVE"
        elif bool(row.entry_condition):
            state = "FORCED_SELLING_ACTIVE"
            start_event = True
            active = True
            consecutive_exhaustion = 0
        else:
            state = "NORMAL"
        states.append(state)
        start_events.append(start_event)
        exhaustion_events.append(exhaustion_event)
    output["state"] = states
    output["pressure_start_event"] = start_events
    output["pressure_exhaustion_event"] = exhaustion_events
    if not set(output["state"].unique()).issubset(EXPECTED_STATES):
        raise ValueError("状态机产生未冻结状态")
    output["target_position_signal"] = np.where(
        output["state"].eq("FORCED_SELLING_ACTIVE"), 0.0, 1.0
    )
    return output

def rolling_prior_percentile(
    values: pd.Series, window: int, minimum_observations: int
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    for index, current in enumerate(numeric):
        if not np.isfinite(current):
            continue
        start = max(0, index - int(window))
        history = numeric[start:index]
        history = history[np.isfinite(history)]
        if len(history) < int(minimum_observations):
            continue
        less = np.count_nonzero(history < current)
        equal = np.count_nonzero(history == current)
        result[index] = (less + 0.5 * equal) / len(history)
    return pd.Series(result, index=values.index, dtype=float)
