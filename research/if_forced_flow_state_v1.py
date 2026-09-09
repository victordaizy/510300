"""510300 IF基差压力、持仓冲击与强制资金流耗竭状态研究V1。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import (
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)
from research.direction_switch_common_v1 import (
    ROOT,
    atomic_json,
    atomic_parquet,
    atomic_text,
    generated_at,
    holm_bonferroni,
    index_cagr,
    normalize_dividends,
    normalize_market,
    prepare_total_return_market,
    required_columns,
    rolling_prior_percentile,
    sha256_file,
)


EXPECTED_STATES = [
    "NORMAL",
    "FORCED_SELLING_ACTIVE",
    "FORCED_SELLING_EXHAUSTED",
]


def load_if_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = config["inputs"]
    contract = config["if_data_contract"]
    daily = pd.read_parquet(ROOT / inputs["if_contract_daily"]["file"])
    expiry = pd.read_parquet(ROOT / inputs["if_contract_expiry"]["file"])
    acquisition = json.loads(
        (ROOT / inputs["if_acquisition_receipt"]["file"]).read_text(
            encoding="utf-8"
        )
    )
    spot = normalize_market(
        pd.read_parquet(ROOT / inputs["if_spot_index"]["file"]), "000300.SH"
    )
    market = normalize_market(
        pd.read_parquet(ROOT / inputs["if_execution_etf"]["file"]), "510300.SH"
    )
    benchmark = pd.read_parquet(ROOT / inputs["if_h00300"]["file"])
    dividends = normalize_dividends(
        pd.read_csv(ROOT / inputs["dividends"]["file"])
    )
    required_columns(
        daily,
        inputs["if_contract_daily"]["required_fields"],
        "中金所IF逐合约日线",
    )
    required_columns(
        expiry,
        [
            "symbol",
            "first_date",
            "last_history_date",
            "expiry_date",
            "expiry_source",
            "active_at_ceiling",
        ],
        "IF到期表",
    )
    if len(daily) != int(inputs["if_contract_daily"]["required_rows"]):
        raise ValueError("IF逐合约行数不匹配")
    if len(expiry) != int(inputs["if_contract_expiry"]["required_rows"]):
        raise ValueError("IF到期表行数不匹配")
    if acquisition.get("status") != inputs["if_acquisition_receipt"][
        "required_status"
    ]:
        raise ValueError("IF官方获取凭证状态不匹配")
    if acquisition.get("performance_returns_read") is not False:
        raise ValueError("IF输入获取阶段读取了绩效收益")

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    daily["symbol"] = daily["symbol"].astype(str)
    daily.sort_values(["date", "symbol"], kind="mergesort", inplace=True)
    daily.reset_index(drop=True, inplace=True)
    if daily.duplicated(["date", "symbol"]).any():
        raise ValueError("IF逐合约存在日期代码重复")
    if not daily["symbol"].str.fullmatch(r"IF\d{4}").all():
        raise ValueError("IF逐合约混入非IF代码")
    for column in ["close", "settle", "volume", "open_interest"]:
        daily[column] = pd.to_numeric(daily[column], errors="raise")
    if (daily[["close", "settle"]] <= 0.0).any().any():
        raise ValueError("IF收盘价或结算价存在非正值")
    if (daily[["volume", "open_interest"]] < 0.0).any().any():
        raise ValueError("IF成交量或持仓量存在负值")
    if "archive_yyyymm" in daily.columns:
        mismatch = daily["archive_yyyymm"].astype(str).ne(
            daily["date"].dt.strftime("%Y%m")
        )
        if mismatch.any():
            raise ValueError("IF历史压缩包月份与交易日漂移")
    if not daily["source_url"].astype(str).str.contains(
        "cffex.com.cn", regex=False
    ).all():
        raise ValueError("IF逐合约来源不是中金所")

    expiry = expiry.copy()
    expiry["symbol"] = expiry["symbol"].astype(str)
    for column in ["first_date", "last_history_date", "expiry_date"]:
        expiry[column] = pd.to_datetime(
            expiry[column], errors="raise"
        ).dt.normalize()
    expiry.sort_values("symbol", kind="mergesort", inplace=True)
    if expiry["symbol"].duplicated().any():
        raise ValueError("IF到期表代码重复")
    if set(expiry["symbol"]) != set(daily["symbol"]):
        raise ValueError("IF逐合约与到期表代码集合不一致")
    active = expiry.loc[expiry["active_at_ceiling"].astype(bool)]
    if not active["expiry_source"].eq(
        "CFFEX_TRADING_PARAMETER_END_TRADING_DAY_AT_CEILING"
    ).all():
        raise ValueError("样本末日存续合约未使用官方到期日")
    historical = expiry.loc[~expiry["active_at_ceiling"].astype(bool)]
    if not historical["expiry_date"].eq(historical["last_history_date"]).all():
        raise ValueError("已到期合约的到期日与最后历史交易日不一致")

    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="raise")
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    if set(benchmark["symbol"].dropna().astype(str).unique()) != {"H00300"}:
        raise ValueError("IF研究的全收益基准代码不是H00300")
    if (benchmark["close"] <= 0.0).any():
        raise ValueError("IF研究的H00300点位存在非正值")

    warmup_start = pd.Timestamp(config["data_scope"]["if_feature_warmup_start"])
    evaluation_end = pd.Timestamp(config["data_scope"]["if_evaluation_end"])
    spot = spot.loc[spot["date"].between(warmup_start, evaluation_end)].reset_index(
        drop=True
    )
    market = market.loc[
        market["date"].between(warmup_start, evaluation_end)
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(warmup_start, evaluation_end)
    ].reset_index(drop=True)
    if not market["date"].equals(spot["date"]):
        missing_spot = sorted(set(market["date"]) - set(spot["date"]))
        raise ValueError(f"沪深300现货缺少510300交易日：{missing_spot[:5]}")
    if sorted(set(market["date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300交易日")
    if int(contract["signal_data_availability_lag_trading_days"]) != 1:
        raise ValueError("无法证明同日发布时间时必须延迟一个交易日")

    joined = daily.merge(
        expiry[
            [
                "symbol",
                "first_date",
                "last_history_date",
                "expiry_date",
                "expiry_source",
                "active_at_ceiling",
            ]
        ],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    joined["calendar_days_to_expiry"] = (
        joined["expiry_date"] - joined["date"]
    ).dt.days
    if (joined["calendar_days_to_expiry"] < 0).any():
        raise ValueError("IF逐合约出现到期日后的行情")
    aggregate_oi = joined.loc[
        joined["calendar_days_to_expiry"].between(1, 190)
    ].groupby("date")["open_interest"].sum()
    aggregate_jump = np.log(aggregate_oi.replace(0.0, np.nan)).diff().abs()
    audit = {
        "status": "PASS_WITH_CONSERVATIVE_ONE_TRADING_DAY_AVAILABILITY_LAG",
        "generated_at": generated_at(),
        "provider": "中国金融期货交易所",
        "official_acquisition_status": acquisition["status"],
        "if_contract_rows": int(len(daily)),
        "if_contract_count": int(daily["symbol"].nunique()),
        "if_trade_dates": int(daily["date"].nunique()),
        "if_first_date": daily["date"].min().date().isoformat(),
        "if_last_date": daily["date"].max().date().isoformat(),
        "expiry_rows": int(len(expiry)),
        "active_contract_rows": int(len(active)),
        "duplicate_date_symbol_rows": 0,
        "negative_open_interest_rows": 0,
        "zero_open_interest_rows": int(daily["open_interest"].eq(0.0).sum()),
        "maximum_one_day_absolute_log_aggregate_oi_change": float(
            aggregate_jump.max()
        ),
        "selected_price_field": contract["selected_price_field"],
        "close_settle_mixing": False,
        "continuous_contract_used": False,
        "official_same_day_publication_time_proven": False,
        "availability_lag_trading_days": 1,
        "availability_decision": contract["lag_reason"],
        "field_reset_or_schema_change_detected": False,
        "source_hashes": {
            name: sha256_file(ROOT / item["file"])
            for name, item in inputs.items()
            if name
            in {
                "if_contract_daily",
                "if_contract_expiry",
                "if_acquisition_receipt",
                "if_spot_index",
                "if_execution_etf",
                "if_h00300",
                "dividends",
            }
        },
        "return_evaluation_allowed": True,
    }
    return {
        "contract_daily": daily,
        "expiry": expiry,
        "joined_contracts": joined,
        "spot": spot,
        "market": prepare_total_return_market(market, dividends),
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


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


def _select_independent_event_indices(
    mask: pd.Series, gap: int
) -> list[int]:
    candidates = np.flatnonzero(mask.to_numpy(dtype=bool)).tolist()
    selected: list[int] = []
    last = -10**9
    for index in candidates:
        if index - last >= int(gap):
            selected.append(int(index))
            last = int(index)
    return selected


def _executable_forward_return(
    frame: pd.DataFrame, signal_index: int, horizon: int
) -> float | None:
    entry_index = signal_index + 1
    exit_index = signal_index + int(horizon)
    if entry_index >= len(frame) or exit_index >= len(frame):
        return None
    entry_open = float(frame.iloc[entry_index]["open"])
    exit_close = float(frame.iloc[exit_index]["close"])
    cash_dividend = float(
        frame.iloc[entry_index + 1 : exit_index + 1][
            "cash_dividend_per_share"
        ].sum()
    )
    return float((exit_close + cash_dividend) / entry_open - 1.0)


def _standardized_mean_differences(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    covariates: list[str],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if events.empty or controls.empty:
        return {column: None for column in covariates}
    for column in covariates:
        event_values = pd.to_numeric(events[column], errors="coerce").dropna()
        control_values = pd.to_numeric(controls[column], errors="coerce").dropna()
        if event_values.empty or control_values.empty:
            result[column] = None
            continue
        pooled = math.sqrt(
            (
                float(event_values.var(ddof=1))
                + float(control_values.var(ddof=1))
            )
            / 2.0
        )
        result[column] = (
            float((event_values.mean() - control_values.mean()) / pooled)
            if pooled > 0.0 and np.isfinite(pooled)
            else 0.0
        )
    return result


def _ordered_block_bootstrap_p_value(
    effects: np.ndarray,
    *,
    expected_direction: str,
    replicates: int,
    block_length: int,
    seed: int,
) -> float | None:
    values = np.asarray(effects, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    generator = np.random.default_rng(seed)
    block = min(max(int(block_length), 1), len(values))
    means = np.empty(int(replicates), dtype=float)
    for replicate in range(int(replicates)):
        sampled: list[int] = []
        while len(sampled) < len(values):
            start = int(generator.integers(0, len(values)))
            sampled.extend((start + offset) % len(values) for offset in range(block))
        means[replicate] = float(values[np.asarray(sampled[: len(values)])].mean())
    if expected_direction == "negative":
        extreme = int(np.count_nonzero(means >= 0.0))
    elif expected_direction == "positive":
        extreme = int(np.count_nonzero(means <= 0.0))
    else:
        raise ValueError("Bootstrap方向必须为negative或positive")
    return float((extreme + 1) / (len(means) + 1))


def _year_effect_share(effects: pd.DataFrame, effect_column: str) -> float | None:
    if effects.empty:
        return None
    yearly = effects.groupby(effects["event_date"].dt.year)[effect_column].sum()
    denominator = float(yearly.abs().sum())
    if denominator <= 0.0:
        return None
    return float(yearly.abs().max() / denominator)


def _leave_one_year_direction_fraction(
    effects: pd.DataFrame, effect_column: str, expected_direction: str
) -> float | None:
    years = sorted(effects["event_date"].dt.year.unique())
    if len(years) < 2:
        return None
    passes: list[bool] = []
    for year in years:
        remaining = effects.loc[effects["event_date"].dt.year.ne(year), effect_column]
        mean = float(remaining.mean()) if not remaining.empty else np.nan
        passes.append(mean < 0.0 if expected_direction == "negative" else mean > 0.0)
    return float(np.mean(passes))


def _match_event_type(
    config: dict[str, Any],
    states: pd.DataFrame,
    *,
    event_type: str,
    event_indices: list[int],
    all_event_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, float | None]]:
    matching = config["matching"]
    horizons = [int(value) for value in config["if_signal"]["fixed_horizons_trading_days"]]
    covariates = list(matching["covariates"])
    rank_columns = [f"rank_{column}" for column in covariates]
    radius = int(matching["event_exclusion_radius_trading_days"])
    control_count = int(matching["controls_per_event"])
    maximum_component = float(
        matching["maximum_absolute_rank_distance_per_covariate"]
    )
    maximum_euclidean = float(matching["maximum_euclidean_rank_distance"])
    blocked_control_indices = {
        index
        for event_index in all_event_indices
        for index in range(
            max(0, event_index - radius),
            min(len(states), event_index + radius + 1),
        )
    }
    event_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for sequence, event_index in enumerate(event_indices, start=1):
        if event_index + max(horizons) >= len(states):
            continue
        event = states.iloc[event_index]
        if event[rank_columns].isna().any():
            continue
        candidates = states.loc[
            states["date"].dt.year.eq(pd.Timestamp(event["date"]).year)
            & states["state"].eq("NORMAL")
            & states["data_eligible"]
        ].copy()
        candidates = candidates.loc[
            ~candidates.index.isin(blocked_control_indices)
            & (candidates.index + max(horizons) < len(states))
        ]
        candidates = candidates.dropna(subset=rank_columns)
        if candidates.empty:
            continue
        difference = candidates[rank_columns].sub(
            event[rank_columns].to_numpy(dtype=float), axis=1
        ).abs()
        component_ok = difference.le(maximum_component).all(axis=1)
        distance = np.sqrt((difference * difference).sum(axis=1))
        eligible = candidates.loc[component_ok & distance.le(maximum_euclidean)].copy()
        eligible["match_distance"] = distance.loc[eligible.index]
        eligible.sort_values(
            ["match_distance", "date"], kind="mergesort", inplace=True
        )
        chosen = eligible.head(control_count)
        if len(chosen) < control_count:
            continue
        event_id = f"{event_type}_{sequence:03d}"
        event_row: dict[str, Any] = {
            "event_id": event_id,
            "event_type": event_type,
            "event_date": pd.Timestamp(event["date"]),
            "feature_source_date": pd.Timestamp(event["feature_source_date"]),
            "basis_residual_percentile": float(event["basis_residual_percentile"]),
            "open_interest_shock_percentile": float(
                event["open_interest_shock_percentile"]
            ),
            "etf_past_3d_total_return": float(event["etf_past_3d_total_return"]),
            **{column: float(event[column]) for column in covariates},
        }
        for horizon in horizons:
            event_return = _executable_forward_return(states, event_index, horizon)
            control_values: list[float] = []
            for control_sequence, (control_index, control) in enumerate(
                chosen.iterrows(), start=1
            ):
                control_return = _executable_forward_return(
                    states, int(control_index), horizon
                )
                if control_return is None:
                    raise ValueError("已匹配控制缺少冻结期限收益")
                control_values.append(control_return)
                if horizon == horizons[0]:
                    control_rows.append(
                        {
                            "event_id": event_id,
                            "event_type": event_type,
                            "event_date": pd.Timestamp(event["date"]),
                            "control_id": f"{event_id}_C{control_sequence:02d}",
                            "control_date": pd.Timestamp(control["date"]),
                            "match_distance": float(control["match_distance"]),
                            **{column: float(control[column]) for column in covariates},
                        }
                    )
            if event_return is None:
                raise ValueError("完整事件意外缺少冻结期限收益")
            control_mean = float(np.mean(control_values))
            event_row[f"event_return_{horizon}d"] = float(event_return)
            event_row[f"control_mean_return_{horizon}d"] = control_mean
            event_row[f"matched_effect_{horizon}d"] = float(
                event_return - control_mean
            )
        event_rows.append(event_row)
    event_frame = pd.DataFrame(event_rows)
    control_frame = pd.DataFrame(control_rows)
    smd = _standardized_mean_differences(event_frame, control_frame, covariates)
    balance_values = [abs(value) for value in smd.values() if value is not None]
    diagnostics = {
        "independent_candidate_events": int(len(event_indices)),
        "matched_complete_events": int(len(event_frame)),
        "calendar_years": (
            sorted(event_frame["event_date"].dt.year.unique().astype(int).tolist())
            if not event_frame.empty
            else []
        ),
        "controls_per_matched_event": control_count,
        "maximum_absolute_standardized_mean_difference": (
            max(balance_values) if balance_values else None
        ),
        "standardized_mean_differences": smd,
    }
    raw_p_values: dict[str, float | None] = {}
    bootstrap = config["mechanism_tests"]["block_bootstrap"]
    expected = "negative" if event_type == "PRESSURE_START" else "positive"
    for horizon in horizons:
        effects = (
            event_frame[f"matched_effect_{horizon}d"].to_numpy(dtype=float)
            if not event_frame.empty
            else np.asarray([], dtype=float)
        )
        raw_p_values[f"{event_type}_{horizon}d"] = _ordered_block_bootstrap_p_value(
            effects,
            expected_direction=expected,
            replicates=int(bootstrap["replicates"]),
            block_length=int(bootstrap["ordered_event_block_length"]),
            seed=int(bootstrap["seed"]) + horizon + (0 if expected == "negative" else 1000),
        )
    return event_frame, control_frame, diagnostics, raw_p_values


def evaluate_mechanism(
    config: dict[str, Any], states: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    signal = config["if_signal"]
    tests = config["mechanism_tests"]
    evaluation_start = pd.Timestamp(config["data_scope"]["if_evaluation_start"])
    evaluation_end = pd.Timestamp(config["data_scope"]["if_evaluation_end"])
    evaluation = states.loc[
        states["date"].between(evaluation_start, evaluation_end)
    ].copy()
    if evaluation.empty:
        raise ValueError("IF机制评估区间为空")
    gap = int(signal["independent_event_gap_trading_days"])
    start_indices = _select_independent_event_indices(
        states["pressure_start_event"] & states["date"].between(evaluation_start, evaluation_end),
        gap,
    )
    exhaustion_indices = _select_independent_event_indices(
        states["pressure_exhaustion_event"]
        & states["date"].between(evaluation_start, evaluation_end),
        gap,
    )
    minimum_events = int(tests["minimum_independent_events_each_direction"])
    if len(start_indices) < minimum_events or len(exhaustion_indices) < minimum_events:
        def insufficient_gate(indices: list[int]) -> dict[str, Any]:
            years = sorted(
                states.iloc[indices]["date"].dt.year.unique().astype(int).tolist()
            ) if indices else []
            gates = {
                "minimum_independent_matched_events": False,
                "minimum_calendar_years": len(years)
                >= int(tests["minimum_calendar_years_each_direction"]),
                "single_year_effect_share": False,
                "leave_one_year_out_direction": False,
                "matching_balance": False,
                "primary_economic_effect": False,
                "holm_adjusted_significance": False,
            }
            return {
                "passed": False,
                "gates": gates,
                "failed_gates": [name for name, passed in gates.items() if not passed],
                "primary_mean_matched_effect": None,
                "maximum_single_year_absolute_effect_share": None,
                "leave_one_year_out_expected_direction_fraction": None,
                "primary_raw_p_value": None,
                "primary_holm_adjusted_p_value": None,
                "independent_candidate_events": len(indices),
                "matched_complete_events": 0,
                "calendar_years": years,
                "controls_per_matched_event": int(
                    config["matching"]["controls_per_event"]
                ),
                "maximum_absolute_standardized_mean_difference": None,
                "standardized_mean_differences": {
                    column: None for column in config["matching"]["covariates"]
                },
                "not_evaluated_reason": "MINIMUM_INDEPENDENT_EVENT_COUNT_FAILED_BEFORE_RETURN_READ",
            }

        start_gate = insufficient_gate(start_indices)
        exhaustion_gate = insufficient_gate(exhaustion_indices)
        empty_events = pd.DataFrame(
            columns=[
                "event_id",
                "event_type",
                "event_date",
                "not_evaluated_reason",
            ]
        )
        empty_controls = pd.DataFrame(
            columns=[
                "event_id",
                "event_type",
                "event_date",
                "control_id",
                "control_date",
                "not_evaluated_reason",
            ]
        )
        report = {
            "status": "REJECTED_FROZEN_INSUFFICIENT_INDEPENDENT_EVENT_SUPPORT_NO_RESCUE",
            "passed": False,
            "return_evaluation": "NOT_ALLOWED",
            "evaluation_start": evaluation_start.date().isoformat(),
            "evaluation_end": evaluation_end.date().isoformat(),
            "raw_state_counts": {
                "pressure_start": int(evaluation["pressure_start_event"].sum()),
                "pressure_exhaustion": int(
                    evaluation["pressure_exhaustion_event"].sum()
                ),
            },
            "independent_event_counts": {
                "pressure_start": len(start_indices),
                "pressure_exhaustion": len(exhaustion_indices),
            },
            "minimum_required_each_direction": minimum_events,
            "pressure_start_gate": start_gate,
            "pressure_exhaustion_gate": exhaustion_gate,
            "raw_one_sided_p_values": {},
            "holm_adjusted_p_values": {},
            "multiple_test_family": "NOT_RUN_EVENT_COUNT_GATE_FAILED",
            "future_return_columns_read": 0,
            "matching_run": False,
            "block_bootstrap_run": False,
            "direction_reversal_allowed": False,
            "fixed_holding_period_fallback_allowed": False,
        }
        return report, empty_events, empty_controls
    all_event_indices = sorted(set(start_indices + exhaustion_indices))
    start_events, start_controls, start_diagnostics, start_p = _match_event_type(
        config,
        states,
        event_type="PRESSURE_START",
        event_indices=start_indices,
        all_event_indices=all_event_indices,
    )
    exhaustion_events, exhaustion_controls, exhaustion_diagnostics, exhaustion_p = (
        _match_event_type(
            config,
            states,
            event_type="PRESSURE_EXHAUSTION",
            event_indices=exhaustion_indices,
            all_event_indices=all_event_indices,
        )
    )
    raw_p = {**start_p, **exhaustion_p}
    adjusted_p = holm_bonferroni(raw_p)
    primary = int(tests["primary_horizon_trading_days"])

    def gate_event_type(
        event_frame: pd.DataFrame,
        diagnostics: dict[str, Any],
        expected: str,
        p_key: str,
    ) -> dict[str, Any]:
        effect_column = f"matched_effect_{primary}d"
        mean_effect = (
            float(event_frame[effect_column].mean()) if not event_frame.empty else None
        )
        year_share = _year_effect_share(event_frame, effect_column)
        loyo = _leave_one_year_direction_fraction(
            event_frame, effect_column, expected
        )
        smd_max = diagnostics["maximum_absolute_standardized_mean_difference"]
        gates = {
            "minimum_independent_matched_events": len(event_frame)
            >= int(tests["minimum_independent_events_each_direction"]),
            "minimum_calendar_years": len(diagnostics["calendar_years"])
            >= int(tests["minimum_calendar_years_each_direction"]),
            "single_year_effect_share": year_share is not None
            and year_share
            <= float(tests["maximum_single_year_absolute_effect_share"]),
            "leave_one_year_out_direction": loyo is not None
            and loyo
            >= float(tests["leave_one_year_out_expected_direction_fraction_min"]),
            "matching_balance": smd_max is not None
            and smd_max
            <= float(
                config["matching"]["maximum_absolute_standardized_mean_difference"]
            ),
            "primary_economic_effect": mean_effect is not None
            and (
                mean_effect <= float(tests["expected_start_effect_maximum"])
                if expected == "negative"
                else mean_effect
                >= float(tests["expected_exhaustion_effect_minimum"])
            ),
            "holm_adjusted_significance": adjusted_p.get(p_key) is not None
            and float(adjusted_p[p_key]) <= float(tests["one_sided_alpha"]),
        }
        return {
            "passed": all(gates.values()),
            "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "primary_mean_matched_effect": mean_effect,
            "maximum_single_year_absolute_effect_share": year_share,
            "leave_one_year_out_expected_direction_fraction": loyo,
            "primary_raw_p_value": raw_p.get(p_key),
            "primary_holm_adjusted_p_value": adjusted_p.get(p_key),
            **diagnostics,
        }

    start_gate = gate_event_type(
        start_events, start_diagnostics, "negative", f"PRESSURE_START_{primary}d"
    )
    exhaustion_gate = gate_event_type(
        exhaustion_events,
        exhaustion_diagnostics,
        "positive",
        f"PRESSURE_EXHAUSTION_{primary}d",
    )
    passed = bool(start_gate["passed"] and exhaustion_gate["passed"])
    status = (
        "PASS_BOTH_MECHANISM_GATES_PORTFOLIO_RUN_ALLOWED_ONCE"
        if passed
        else "REJECTED_FROZEN_MECHANISM_GATE_FAILED_NO_RESCUE"
    )
    event_frame = pd.concat(
        [start_events, exhaustion_events], ignore_index=True, sort=False
    )
    control_frame = pd.concat(
        [start_controls, exhaustion_controls], ignore_index=True, sort=False
    )
    report = {
        "status": status,
        "passed": passed,
        "evaluation_start": evaluation_start.date().isoformat(),
        "evaluation_end": evaluation_end.date().isoformat(),
        "raw_state_counts": {
            "pressure_start": int(
                evaluation["pressure_start_event"].sum()
            ),
            "pressure_exhaustion": int(
                evaluation["pressure_exhaustion_event"].sum()
            ),
        },
        "independent_event_counts": {
            "pressure_start": len(start_indices),
            "pressure_exhaustion": len(exhaustion_indices),
        },
        "pressure_start_gate": start_gate,
        "pressure_exhaustion_gate": exhaustion_gate,
        "raw_one_sided_p_values": raw_p,
        "holm_adjusted_p_values": adjusted_p,
        "multiple_test_family": "2_EVENT_TYPES_X_4_HORIZONS",
        "direction_reversal_allowed": False,
        "fixed_holding_period_fallback_allowed": False,
    }
    return report, event_frame, control_frame


def _portfolio_costs(config: dict[str, Any], double_cost: bool) -> BacktestCosts:
    execution = config["execution"]
    return BacktestCosts(
        commission_rate=float(
            execution[
                "double_cost_commission_rate_per_leg"
                if double_cost
                else "commission_rate_per_leg"
            ]
        ),
        minimum_commission_cny=float(
            execution["minimum_commission_cny_per_leg"]
        ),
        stamp_duty_rate=float(execution["stamp_duty_rate"]),
        slippage_bps=float(
            execution[
                "double_cost_slippage_bps_per_leg"
                if double_cost
                else "slippage_bps_per_leg"
            ]
        ),
        lot_size=int(execution["lot_size_shares"]),
        cash_annual_rate=float(execution["cash_annual_rate"]),
    )


def _portfolio_concentration(
    strategy: pd.DataFrame, buy_hold: pd.DataFrame
) -> dict[str, Any]:
    aligned = strategy[["date", "daily_return"]].merge(
        buy_hold[["date", "daily_return"]].rename(
            columns={"daily_return": "buy_hold_daily_return"}
        ),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    aligned["log_excess"] = np.log1p(aligned["daily_return"]) - np.log1p(
        aligned["buy_hold_daily_return"]
    )
    aligned["year"] = aligned["date"].dt.year.astype(int)
    yearly = aligned.groupby("year")["log_excess"].sum()
    denominator = float(yearly.abs().sum())
    maximum_share = (
        float(yearly.abs().max() / denominator) if denominator > 0.0 else None
    )
    years = sorted(yearly.index.tolist())
    leave_one_out: dict[str, float] = {}
    for year in years:
        value = float(aligned.loc[aligned["year"].ne(year), "log_excess"].sum())
        leave_one_out[str(year)] = value
    direction_fraction = (
        float(np.mean([value > 0.0 for value in leave_one_out.values()]))
        if leave_one_out
        else None
    )
    return {
        "yearly_log_excess": {str(year): float(value) for year, value in yearly.items()},
        "maximum_single_year_absolute_log_excess_share": maximum_share,
        "leave_one_year_out_log_excess": leave_one_out,
        "leave_one_year_out_positive_fraction": direction_fraction,
    }


def run_frozen_portfolio_once(
    config: dict[str, Any],
    inputs: dict[str, Any],
    states: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    gate = config["portfolio_gate"]
    execution = config["execution"]
    start = pd.Timestamp(config["data_scope"]["if_evaluation_start"])
    end = pd.Timestamp(config["data_scope"]["if_evaluation_end"])
    market = inputs["market"].copy().sort_values("date", kind="mergesort")
    start_positions = market.index[market["date"].eq(start)].tolist()
    if len(start_positions) != 1 or start_positions[0] == 0:
        raise ValueError("IF组合回测需要评估起点前一个交易日")
    warmup_start = pd.Timestamp(market.iloc[start_positions[0] - 1]["date"])
    targets = states[["date", "target_position_signal"]].rename(
        columns={"target_position_signal": "target_position"}
    )
    targets["signal_reason"] = "冻结IF强制资金流状态"
    buy_hold_targets = targets[["date"]].copy()
    buy_hold_targets["target_position"] = 1.0
    buy_hold_targets["signal_reason"] = "冻结510300含分红买入持有"
    initial = float(execution["initial_capital_cny"])
    ledger, trades = run_long_cash_backtest(
        market,
        inputs["dividends"],
        targets,
        initial,
        _portfolio_costs(config, False),
        warmup_start,
        end,
    )
    buy_hold_ledger, buy_hold_trades = run_long_cash_backtest(
        market,
        inputs["dividends"],
        buy_hold_targets,
        initial,
        _portfolio_costs(config, False),
        warmup_start,
        end,
    )
    double_ledger, double_trades = run_long_cash_backtest(
        market,
        inputs["dividends"],
        targets,
        initial,
        _portfolio_costs(config, True),
        warmup_start,
        end,
    )
    ledger = ledger.loc[ledger["date"].ge(start)].reset_index(drop=True)
    trades = trades.loc[trades["date"].ge(start)].reset_index(drop=True)
    buy_hold_ledger = buy_hold_ledger.loc[
        buy_hold_ledger["date"].ge(start)
    ].reset_index(drop=True)
    buy_hold_trades = buy_hold_trades.loc[
        buy_hold_trades["date"].ge(start)
    ].reset_index(drop=True)
    double_ledger = double_ledger.loc[
        double_ledger["date"].ge(start)
    ].reset_index(drop=True)
    double_trades = double_trades.loc[
        double_trades["date"].ge(start)
    ].reset_index(drop=True)
    strategy_summary = summarize_backtest(ledger, trades, initial)
    buy_hold_summary = summarize_backtest(
        buy_hold_ledger, buy_hold_trades, initial
    )
    double_summary = summarize_backtest(double_ledger, double_trades, initial)
    benchmark_cagr = index_cagr(
        inputs["benchmark"]["date"],
        inputs["benchmark"]["close"],
        start,
        end,
    )
    concentration = _portfolio_concentration(ledger, buy_hold_ledger)
    drawdown_denominator = abs(float(buy_hold_summary["max_drawdown"]))
    drawdown_ratio = (
        abs(float(strategy_summary["max_drawdown"])) / drawdown_denominator
        if drawdown_denominator > 0.0
        else None
    )
    hard = gate["hard_gates"]
    gates = {
        "net_sharpe": strategy_summary["sharpe_zero_cash_rate"] is not None
        and strategy_summary["sharpe_zero_cash_rate"]
        >= float(hard["net_sharpe_min"]),
        "annualized_excess_vs_510300_tr": strategy_summary["cagr"]
        - buy_hold_summary["cagr"]
        > 0.0,
        "annualized_excess_vs_h00300": strategy_summary["cagr"]
        - benchmark_cagr
        > 0.0,
        "max_drawdown_ratio": drawdown_ratio is not None
        and drawdown_ratio <= float(hard["max_drawdown_ratio_vs_510300_max"]),
        "double_cost_net_sharpe": double_summary["sharpe_zero_cash_rate"]
        is not None
        and double_summary["sharpe_zero_cash_rate"]
        >= float(hard["double_cost_net_sharpe_min"]),
        "leave_one_year_out_direction": concentration[
            "leave_one_year_out_positive_fraction"
        ]
        is not None
        and concentration["leave_one_year_out_positive_fraction"]
        >= float(hard["leave_one_year_out_expected_direction_fraction_min"]),
        "single_year_concentration": concentration[
            "maximum_single_year_absolute_log_excess_share"
        ]
        is not None
        and concentration["maximum_single_year_absolute_log_excess_share"]
        <= float(hard["maximum_single_year_absolute_log_excess_share"]),
    }
    passed = all(gates.values())
    status = gate["success_state"] if passed else gate["failure_state"]
    report = {
        "status": status,
        "passed": passed,
        "strategy": strategy_summary,
        "buy_hold_510300_tr": buy_hold_summary,
        "h00300_cagr": benchmark_cagr,
        "double_cost_strategy": double_summary,
        "annualized_excess_vs_510300_tr": strategy_summary["cagr"]
        - buy_hold_summary["cagr"],
        "annualized_excess_vs_h00300": strategy_summary["cagr"] - benchmark_cagr,
        "max_drawdown_ratio_vs_510300": drawdown_ratio,
        "concentration": concentration,
        "gates": gates,
        "failed_gates": [name for name, passed_gate in gates.items() if not passed_gate],
    }
    return report, ledger, trades


def render_if_report(report: dict[str, Any]) -> str:
    mechanism = report["mechanism"]
    lines = [
        "# 510300 IF强制资金流状态 V1",
        "",
        f"最终状态：`{report['status']}`",
        "",
        "研究阶段：`DISCOVERY_ONLY`；交易资产边界仅为510300或人民币现金。",
        "",
        "## 数据门",
        "",
        f"- 状态：`{report['data_audit']['status']}`",
        f"- IF逐合约：{report['data_audit']['if_contract_rows']}行，{report['data_audit']['if_contract_count']}个合约，{report['data_audit']['if_first_date']}至{report['data_audit']['if_last_date']}。",
        "- 未使用连续合约；收盘价与结算价未混用；因同日官方发布时间不能证明，全部IF日数据延迟一个交易日进入信号。",
        "",
        "## 机制门",
        "",
        f"状态：`{mechanism['status']}`",
        "",
        "| 事件 | 独立候选 | 完整匹配 | 覆盖年份 | 10日匹配效应 | Holm校正p值 | 通过 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("压力开始", "pressure_start_gate"),
        ("压力耗竭", "pressure_exhaustion_gate"),
    ]:
        item = mechanism[key]
        effect = item["primary_mean_matched_effect"]
        adjusted = item["primary_holm_adjusted_p_value"]
        effect_text = "无" if effect is None else f"{effect:.4%}"
        adjusted_text = "无" if adjusted is None else f"{adjusted:.4f}"
        lines.append(
            f"| {label} | {item['independent_candidate_events']} | {item['matched_complete_events']} | {len(item['calendar_years'])} | {effect_text} | {adjusted_text} | {'是' if item['passed'] else '否'} |"
        )
    lines.extend(["", "## 组合层", ""])
    portfolio = report["portfolio"]
    if portfolio["status"].startswith("NOT_ALLOWED"):
        lines.append(
            "机制门未全部通过，冻结组合回测未运行；`NOT_ALLOWED`不是零收益，也不是亏损。"
        )
    else:
        lines.extend(
            [
                f"- 状态：`{portfolio['status']}`",
                f"- 净夏普率：{portfolio['strategy']['sharpe_zero_cash_rate']:.4f}",
                f"- 相对510300含分红买入持有年化超额：{portfolio['annualized_excess_vs_510300_tr']:.4%}",
                f"- 相对H00300年化超额：{portfolio['annualized_excess_vs_h00300']:.4%}",
                f"- 最大回撤比：{portfolio['max_drawdown_ratio_vs_510300']:.4f}",
                f"- 双倍成本净夏普率：{portfolio['double_cost_strategy']['sharpe_zero_cash_rate']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "任何失败均冻结为不救援；不允许改方向、分位数、期限桶、退出条件，或加入宏观、估值、趋势、期权、北向、PCF/IOPV和广度。",
            "",
            "`POSITION_MAPPING_ENABLED=false`，`ORDER_GENERATION_ENABLED=false`，`LIVE_TRADING_AUTHORIZED=false`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_if_artifacts(
    config: dict[str, Any]
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    inputs, data_audit = load_if_inputs(config)
    eligible_contracts, features, feature_audit = build_if_features(config, inputs)
    states = build_state_machine(config, features)
    data_audit["feature_audit"] = feature_audit
    data_audit["eligible_contract_rows_for_basis"] = int(len(eligible_contracts))
    atomic_json(data_audit, ROOT / artifacts["if_data_audit_json"])
    atomic_parquet(states, ROOT / artifacts["if_features"])
    mechanism, events, controls = evaluate_mechanism(config, states)
    atomic_parquet(events, ROOT / artifacts["if_events"])
    atomic_parquet(controls, ROOT / artifacts["if_controls"])
    if mechanism["passed"]:
        portfolio, ledger, trades = run_frozen_portfolio_once(
            config, inputs, states
        )
        atomic_parquet(ledger, ROOT / artifacts["if_portfolio_ledger"])
        atomic_parquet(trades, ROOT / artifacts["if_portfolio_trades"])
        status = portfolio["status"]
    else:
        portfolio = {
            "status": "NOT_ALLOWED_MECHANISM_GATE_FAILED",
            "return_evaluation": "NOT_ALLOWED",
            "reason": "压力开始与压力耗竭两个机制门未全部通过",
        }
        status = mechanism["status"]
    report = {
        "model_id": config["if_signal"]["model_id"],
        "status": status,
        "generated_at": generated_at(),
        "research_stage": config["protocol"]["research_stage"],
        "scope": {
            "information_assets": "CFFEX_IF_ALL_ACTUAL_CONTRACTS_AND_000300",
            "trade_assets": config["protocol"]["trade_assets"],
            "evaluation_start": config["data_scope"]["if_evaluation_start"],
            "evaluation_end": config["data_scope"]["if_evaluation_end"],
            "signal_clock": "AFTER_510300_CLOSE_USING_IF_DATA_LAGGED_ONE_TRADING_DAY",
            "execution_clock": "NEXT_TRADING_DAY_510300_OPEN",
        },
        "data_audit": data_audit,
        "mechanism": mechanism,
        "portfolio": portfolio,
        "predecessor_if_family_status": config["protocol"][
            "predecessor_if_family_status"
        ],
        "predecessor_if_family_reopened": False,
        "new_increment": config["protocol"]["orthogonal_increment"],
        "no_parameter_rescue": True,
        "position_mapping_enabled": False,
        "paper_or_shadow_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
    }
    atomic_json(report, ROOT / artifacts["if_result_json"])
    atomic_text(render_if_report(report), ROOT / artifacts["if_result_markdown"])
    output_paths = [
        artifacts["if_data_audit_json"],
        artifacts["if_features"],
        artifacts["if_events"],
        artifacts["if_controls"],
        artifacts["if_result_json"],
        artifacts["if_result_markdown"],
    ]
    if mechanism["passed"]:
        output_paths.extend(
            [artifacts["if_portfolio_ledger"], artifacts["if_portfolio_trades"]]
        )
    return {
        "report": report,
        "artifact_hashes": {
            relative: sha256_file(ROOT / relative) for relative in output_paths
        },
    }
