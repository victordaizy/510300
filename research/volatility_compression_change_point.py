"""510300波动压缩事件、未来变盘结果与匹配对照。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from research.graph_regime_martin_turtle_v2 import add_point_in_time_adjusted_ohlc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "volatility_compression_change_point_v1.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "510300_VOLATILITY_COMPRESSION_CHANGE_POINT_V1":
        raise ValueError("波动压缩项目ID错误")
    return config


def load_inputs(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(root / config["data"]["market_file"])
    dividends = pd.read_csv(root / config["data"]["dividend_file"])
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dividends = dividends.sort_values("ex_date").reset_index(drop=True)
    return market, dividends


def build_features(
    market: pd.DataFrame, dividends: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    data = add_point_in_time_adjusted_ohlc(market, dividends)
    close = data["adjusted_close"].astype(float)
    data["daily_amplitude"] = (
        data["adjusted_high"] - data["adjusted_low"]
    ) / close.shift(1)
    data["mean_amplitude_20"] = data["daily_amplitude"].rolling(
        20, min_periods=20
    ).mean()
    data["prior_mean_amplitude_20"] = data["daily_amplitude"].shift(20).rolling(
        20, min_periods=20
    ).mean()
    log_return = np.log(close / close.shift(1))
    annualization = float(
        config["definitions"]["adaptive_rv_compression"]["annualization_days"]
    )
    for window in (5, 20, 60):
        data[f"rv{window}"] = log_return.rolling(window, min_periods=window).std(
            ddof=1
        ) * np.sqrt(annualization)
    adaptive = config["definitions"]["adaptive_rv_compression"]
    data["rv20_trailing_q20"] = data["rv20"].shift(1).rolling(
        int(adaptive["trailing_quantile_window"]),
        min_periods=int(adaptive["trailing_quantile_minimum_observations"]),
    ).quantile(float(adaptive["rv20_quantile"]))
    literal = config["definitions"]["literal_monthly_compression"]
    data["literal_condition"] = (
        data["mean_amplitude_20"].between(
            float(literal["current_mean_amplitude_minimum"]),
            float(literal["current_mean_amplitude_maximum"]),
            inclusive="both",
        )
        & data["prior_mean_amplitude_20"].between(
            float(literal["prior_mean_amplitude_minimum"]),
            float(literal["prior_mean_amplitude_maximum"]),
            inclusive="both",
        )
    )
    data["adaptive_condition"] = (
        data["rv5"].lt(data["rv20"])
        & data["rv20"].lt(data["rv60"])
        & data["rv20"].le(data["rv20_trailing_q20"])
    )
    data["prior20_high"] = data["adjusted_high"].shift(1).rolling(
        20, min_periods=20
    ).max()
    data["prior20_low"] = data["adjusted_low"].shift(1).rolling(
        20, min_periods=20
    ).min()
    data["past20_return"] = close / close.shift(20) - 1.0
    return data


def select_event_indices(
    condition: pd.Series, *, cooldown: int
) -> list[int]:
    values = condition.fillna(False).to_numpy(bool)
    events: list[int] = []
    last_event = -10**9
    for index, active in enumerate(values):
        first_true = active and (index == 0 or not values[index - 1])
        if first_true and index - last_event > cooldown:
            events.append(index)
            last_event = index
    return events


def _first_breakout(
    future: pd.DataFrame, upper: float, lower: float
) -> tuple[str, pd.Timestamp | None, int | None]:
    for offset, (_, row) in enumerate(future.iterrows(), start=1):
        if row["adjusted_close"] > upper:
            return "UP", row["date"], offset
        if row["adjusted_close"] < lower:
            return "DOWN", row["date"], offset
    return "NONE", None, None


def build_events(
    features: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    cooldown = int(config["definitions"]["event_selection"]["cooldown_trading_days"])
    horizons = [int(item) for item in config["outcomes"]["forward_horizons_trading_days"]]
    rules = {
        "LITERAL": "literal_condition",
        "ADAPTIVE": "adaptive_condition",
    }
    rows: list[dict[str, Any]] = []
    for rule_id, column in rules.items():
        indices = select_event_indices(features[column], cooldown=cooldown)
        for number, index in enumerate(indices, start=1):
            row = features.loc[index]
            event: dict[str, Any] = {
                "event_id": f"{rule_id[0]}{number:04d}",
                "rule_id": rule_id,
                "event_index": index,
                "event_date": row["date"],
                "adjusted_close": float(row["adjusted_close"]),
                "mean_amplitude_20": float(row["mean_amplitude_20"]),
                "prior_mean_amplitude_20": float(row["prior_mean_amplitude_20"]),
                "rv5": float(row["rv5"]),
                "rv20": float(row["rv20"]),
                "rv60": float(row["rv60"]),
                "rv20_trailing_q20": float(row["rv20_trailing_q20"])
                if pd.notna(row["rv20_trailing_q20"])
                else np.nan,
                "past20_return": float(row["past20_return"]),
                "prior20_high": float(row["prior20_high"]),
                "prior20_low": float(row["prior20_low"]),
            }
            for horizon in horizons:
                event[f"mature_{horizon}d"] = index + horizon < len(features)
                if not event[f"mature_{horizon}d"]:
                    for name in (
                        "future_mean_amplitude",
                        "future_amplitude_ratio",
                        "future_rv",
                        "future_return",
                        "future_absolute_return",
                        "maximum_up_excursion",
                        "maximum_down_excursion",
                    ):
                        event[f"{name}_{horizon}d"] = np.nan
                    continue
                future = features.iloc[index + 1 : index + horizon + 1]
                future_amp = float(future["daily_amplitude"].mean())
                future_returns = np.log(
                    future["adjusted_close"] / future["adjusted_close"].shift(1)
                )
                first_previous = float(features.loc[index, "adjusted_close"])
                log_path = np.log(
                    pd.concat(
                        [pd.Series([first_previous]), future["adjusted_close"].reset_index(drop=True)],
                        ignore_index=True,
                    )
                ).diff().dropna()
                end_close = float(future.iloc[-1]["adjusted_close"])
                signed_return = end_close / first_previous - 1.0
                event[f"future_mean_amplitude_{horizon}d"] = future_amp
                event[f"future_amplitude_ratio_{horizon}d"] = (
                    future_amp / float(row["mean_amplitude_20"])
                )
                event[f"future_rv_{horizon}d"] = float(
                    log_path.std(ddof=1) * np.sqrt(242)
                )
                event[f"future_return_{horizon}d"] = signed_return
                event[f"future_absolute_return_{horizon}d"] = abs(signed_return)
                event[f"maximum_up_excursion_{horizon}d"] = float(
                    future["adjusted_high"].max() / first_previous - 1.0
                )
                event[f"maximum_down_excursion_{horizon}d"] = float(
                    future["adjusted_low"].min() / first_previous - 1.0
                )
            future20 = features.iloc[index + 1 : min(index + 21, len(features))]
            if len(future20) == 20:
                label, breakout_date, breakout_offset = _first_breakout(
                    future20,
                    float(row["prior20_high"]),
                    float(row["prior20_low"]),
                )
                event["first_breakout_20d"] = label
                event["first_breakout_date"] = breakout_date
                event["first_breakout_offset_days"] = breakout_offset
                future_return = float(event["future_return_20d"])
                past_return = float(event["past20_return"])
                event["direction_reversal_20d"] = (
                    np.sign(future_return) == -np.sign(past_return)
                    if future_return != 0.0 and past_return != 0.0
                    else False
                )
            else:
                event["first_breakout_20d"] = None
                event["first_breakout_date"] = None
                event["first_breakout_offset_days"] = None
                event["direction_reversal_20d"] = None
            rows.append(event)
    return pd.DataFrame(rows).sort_values(["event_date", "rule_id"]).reset_index(drop=True)


def _anchor_metrics(features: pd.DataFrame, index: int) -> dict[str, float]:
    future = features.iloc[index + 1 : index + 21]
    start = float(features.loc[index, "adjusted_close"])
    end = float(future.iloc[-1]["adjusted_close"])
    return {
        "amplitude_ratio": float(future["daily_amplitude"].mean())
        / float(features.loc[index, "mean_amplitude_20"]),
        "absolute_return": abs(end / start - 1.0),
        "signed_return": end / start - 1.0,
    }


def matched_control(
    features: pd.DataFrame,
    events: pd.DataFrame,
    rule_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    control = config["matched_control"]
    tolerance = float(control["current_amplitude_relative_tolerance"])
    exclusion = int(control["exclude_within_event_trading_days"])
    minimum_candidates = int(control["minimum_candidates_per_event"])
    repetitions = int(control["repetitions"])
    primary = events.loc[
        events["rule_id"].eq(rule_id) & events["mature_20d"].astype(bool)
    ].copy()
    if primary.empty:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "NO_MATURE_EVENTS"}
    eligible = features.index[
        features["mean_amplitude_20"].notna()
        & features.index.to_series().le(len(features) - 21)
    ].to_numpy(int)
    event_indices = events.loc[events["rule_id"].eq(rule_id), "event_index"].to_numpy(int)
    pools: list[np.ndarray] = []
    candidate_counts: list[int] = []
    for event in primary.itertuples(index=False):
        target = float(event.mean_amplitude_20)
        candidates = []
        for index in eligible:
            if any(abs(index - event_index) <= exclusion for event_index in event_indices):
                continue
            current = float(features.loc[index, "mean_amplitude_20"])
            if target * (1.0 - tolerance) <= current <= target * (1.0 + tolerance):
                candidates.append(index)
        pool = np.asarray(candidates, dtype=int)
        candidate_counts.append(len(pool))
        pools.append(pool)
    if min(candidate_counts) < minimum_candidates:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "MATCH_POOL_TOO_SMALL",
            "candidate_counts": candidate_counts,
        }
    observed = {
        "amplitude_ratio": float(primary["future_amplitude_ratio_20d"].mean()),
        "absolute_return": float(primary["future_absolute_return_20d"].mean()),
        "signed_return": float(primary["future_return_20d"].mean()),
    }
    cache = {index: _anchor_metrics(features, index) for pool in pools for index in pool}
    rng = np.random.default_rng(int(control["random_seed"]) + (0 if rule_id == "LITERAL" else 1))
    placebo = {key: np.empty(repetitions, dtype=float) for key in observed}
    for repetition in range(repetitions):
        selected = [int(rng.choice(pool)) for pool in pools]
        for key in observed:
            placebo[key][repetition] = np.mean([cache[index][key] for index in selected])
    amplitude_values = placebo["amplitude_ratio"]
    amplitude_percentile = float(
        (np.sum(amplitude_values < observed["amplitude_ratio"]) + 0.5 * np.sum(amplitude_values == observed["amplitude_ratio"]))
        / repetitions
    )
    signed_values = placebo["signed_return"]
    lower = (np.sum(signed_values <= observed["signed_return"]) + 1) / (repetitions + 1)
    upper = (np.sum(signed_values >= observed["signed_return"]) + 1) / (repetitions + 1)
    return {
        "status": "PASS",
        "mature_event_count": int(len(primary)),
        "candidate_counts": candidate_counts,
        "observed_means": observed,
        "placebo_mean_amplitude_ratio": float(amplitude_values.mean()),
        "amplitude_ratio_percentile": amplitude_percentile,
        "absolute_return_percentile": float(
            (np.sum(placebo["absolute_return"] < observed["absolute_return"]) + 0.5 * np.sum(placebo["absolute_return"] == observed["absolute_return"]))
            / repetitions
        ),
        "signed_return_two_sided_p": float(min(1.0, 2.0 * min(lower, upper))),
    }


def summarize_rule(
    events: pd.DataFrame,
    rule_id: str,
    matched: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = events.loc[
        events["rule_id"].eq(rule_id) & events["mature_20d"].astype(bool)
    ].copy()
    if primary.empty:
        return {
            "event_count": int(events["rule_id"].eq(rule_id).sum()),
            "mature_20d_event_count": 0,
            "decision": "INSUFFICIENT_EVIDENCE",
            "matched_control": matched,
        }
    ratio = primary["future_amplitude_ratio_20d"]
    breakout_counts = primary["first_breakout_20d"].value_counts().to_dict()
    gates = config["interpretation_gates"]["volatility_preparation_supported_if"]
    enough = len(primary) >= 10
    matched_pass = matched.get("status") == "PASS"
    volatility_gates = {
        "minimum_10_events": enough,
        "median_ratio_above_1": float(ratio.median())
        > float(gates["median_future_amplitude_ratio_20d_strictly_above"]),
        "expansion_frequency": float(ratio.gt(1.0).mean())
        >= float(gates["expansion_frequency_minimum"]),
        "matched_percentile": matched_pass
        and float(matched["amplitude_ratio_percentile"])
        >= float(gates["matched_control_percentile_minimum"]),
    }
    volatility_supported = all(volatility_gates.values())
    direction_supported = matched_pass and float(
        matched["signed_return_two_sided_p"]
    ) <= float(
        config["interpretation_gates"]["directional_signal_supported_if"][
            "matched_control_signed_return_two_sided_p_maximum"
        ]
    )
    horizons: dict[str, Any] = {}
    for horizon in config["outcomes"]["forward_horizons_trading_days"]:
        mature = events.loc[
            events["rule_id"].eq(rule_id)
            & events[f"mature_{horizon}d"].astype(bool)
        ]
        horizons[str(horizon)] = {
            "mature_events": int(len(mature)),
            "median_amplitude_ratio": float(
                mature[f"future_amplitude_ratio_{horizon}d"].median()
            )
            if len(mature)
            else None,
            "expansion_frequency": float(
                mature[f"future_amplitude_ratio_{horizon}d"].gt(1.0).mean()
            )
            if len(mature)
            else None,
            "large_expansion_frequency": float(
                mature[f"future_amplitude_ratio_{horizon}d"].ge(1.5).mean()
            )
            if len(mature)
            else None,
            "mean_signed_return": float(mature[f"future_return_{horizon}d"].mean())
            if len(mature)
            else None,
            "median_absolute_return": float(
                mature[f"future_absolute_return_{horizon}d"].median()
            )
            if len(mature)
            else None,
        }
    return {
        "event_count": int(events["rule_id"].eq(rule_id).sum()),
        "mature_20d_event_count": int(len(primary)),
        "current_amplitude_20_mean": float(primary["mean_amplitude_20"].mean()),
        "future_amplitude_ratio_20_median": float(ratio.median()),
        "future_amplitude_ratio_20_mean": float(ratio.mean()),
        "expansion_frequency_20d": float(ratio.gt(1.0).mean()),
        "large_expansion_frequency_20d": float(ratio.ge(1.5).mean()),
        "mean_future_return_20d": float(primary["future_return_20d"].mean()),
        "median_absolute_return_20d": float(primary["future_absolute_return_20d"].median()),
        "mean_maximum_up_excursion_20d": float(primary["maximum_up_excursion_20d"].mean()),
        "mean_maximum_down_excursion_20d": float(primary["maximum_down_excursion_20d"].mean()),
        "first_breakout_counts_20d": breakout_counts,
        "direction_reversal_frequency_20d": float(primary["direction_reversal_20d"].mean()),
        "matched_control": matched,
        "volatility_preparation_gates": volatility_gates,
        "volatility_preparation_supported": volatility_supported,
        "directional_signal_supported": direction_supported,
        "decision": "VOLATILITY_PREPARATION_SUPPORTED"
        if volatility_supported
        else "INSUFFICIENT_EVIDENCE"
        if not enough
        else "VOLATILITY_PREPARATION_NOT_SUPPORTED",
        "horizons": horizons,
    }


def recent_state(
    features: pd.DataFrame, events: pd.DataFrame
) -> dict[str, Any]:
    row = features.iloc[-1]
    state: dict[str, Any] = {
        "date": str(row["date"].date()),
        "mean_amplitude_20": float(row["mean_amplitude_20"]),
        "prior_mean_amplitude_20": float(row["prior_mean_amplitude_20"]),
        "rv5": float(row["rv5"]),
        "rv20": float(row["rv20"]),
        "rv60": float(row["rv60"]),
        "rv20_trailing_q20": float(row["rv20_trailing_q20"]),
        "literal_condition": bool(row["literal_condition"]),
        "adaptive_condition": bool(row["adaptive_condition"]),
    }
    for rule_id in ("LITERAL", "ADAPTIVE"):
        subset = events.loc[events["rule_id"].eq(rule_id)]
        state[f"last_{rule_id.lower()}_event_date"] = (
            str(pd.Timestamp(subset.iloc[-1]["event_date"]).date()) if not subset.empty else None
        )
    return state
