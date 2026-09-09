"""VAL01原始EY五年分位的T+1未来收益标签与受污染历史预测诊断。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "val01_raw_ey_5y_v1_forward_evaluation.yaml"

SIGNAL_VALUE_COLUMNS_FORBIDDEN_IN_LABELS = {
    "weighted_raw_earnings_yield",
    "raw_ey_percentile_60m",
    "percentile_window_observations",
    "percentile_ready",
    "percentile_status",
    "target_position",
    "target_shares",
    "order_quantity",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if not protocol.get("forward_label_generation_enabled"):
        raise ValueError("未来收益标签阶段必须显式启用标签生成")
    if not protocol.get("predictive_ic_calculation_enabled"):
        raise ValueError("未来收益标签阶段必须显式启用预测IC诊断")
    forbidden = (
        "historical_strategy_return_calculation_enabled",
        "historical_position_mapping_enabled",
        "current_position_mapping_enabled",
        "target_share_generation_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    )
    enabled = [name for name in forbidden if protocol.get(name)]
    if enabled:
        raise ValueError(f"预测屏幕禁止启用：{enabled}")
    if config["true_forward"]["start"] is not None:
        raise ValueError("受污染历史不得伪造真实前向起点")
    return config


def verify_input_hashes(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "dataset": name,
                "file": contract["file"],
                "expected_sha256": contract["sha256"],
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "BLOCKED_HASH_DRIFT",
        "rows": rows,
    }


def _normalize_market(data: pd.DataFrame, columns: set[str], label: str) -> pd.DataFrame:
    if missing := columns.difference(data.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    numeric = columns.difference({"date"})
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["date", *sorted(numeric)]].isna().any().any():
        raise ValueError(f"{label}存在空日期或数值")
    if result.duplicated("date").any():
        raise ValueError(f"{label}存在重复交易日")
    if result[list(numeric)].le(0).any().any():
        raise ValueError(f"{label}存在非正价格")
    return result.sort_values("date").reset_index(drop=True)


def build_common_market(
    etf: pd.DataFrame,
    price_index: pd.DataFrame,
    total_return_index: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    etf_data = _normalize_market(etf, {"date", "open", "close"}, "510300日线")
    price_data = _normalize_market(price_index, {"date", "open", "close"}, "000300日线")
    total_data = _normalize_market(total_return_index, {"date", "close"}, "H00300日线")
    etf_dates = set(etf_data["date"])
    price_dates = set(price_data["date"])
    total_dates = set(total_data["date"])
    common_dates = etf_dates & price_dates & total_dates
    market = etf_data[["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    market = market.merge(
        price_data[["date", "open", "close"]].rename(
            columns={"open": "price_index_open", "close": "price_index_close"}
        ),
        on="date",
        validate="one_to_one",
    )
    market = market.merge(
        total_data[["date", "close"]].rename(columns={"close": "h00300_close"}),
        on="date",
        validate="one_to_one",
    )
    market["h00300_synthetic_open"] = (
        market["h00300_close"] * market["price_index_open"] / market["price_index_close"]
    )
    market["h00300_synthetic_close"] = (
        market["price_index_close"]
        * market["h00300_close"]
        / market["price_index_close"]
    )
    cutoff = pd.Timestamp(config["protocol"]["label_data_cutoff"])
    market = market.loc[market["date"].le(cutoff)].reset_index(drop=True)
    close_error = float(
        (market["h00300_synthetic_close"] - market["h00300_close"]).abs().max()
    )
    audit = {
        "etf_rows": int(len(etf_data)),
        "price_index_rows": int(len(price_data)),
        "total_return_index_rows": int(len(total_data)),
        "common_calendar_rows_full_files": int(len(common_dates)),
        "common_calendar_rows_through_cutoff": int(len(market)),
        "common_calendar_first_date": str(market["date"].min().date()),
        "common_calendar_last_date": str(market["date"].max().date()),
        "etf_only_date_count": int(len(etf_dates - common_dates)),
        "price_index_only_date_count": int(len(price_dates - common_dates)),
        "h00300_only_date_count": int(len(total_dates - common_dates)),
        "h00300_only_dates": [str(date.date()) for date in sorted(total_dates - common_dates)],
        "synthetic_total_return_close_maximum_error": close_error,
        "status": "PASS" if close_error <= 1e-10 and len(market) > 0 else "BLOCKED",
    }
    return market, audit


def _normalize_dividends(dividends: pd.DataFrame, coverage: dict[str, Any]) -> pd.DataFrame:
    required = {"symbol", "ex_date", "payment_date", "cash_dividend_per_share"}
    if missing := required.difference(dividends.columns):
        raise ValueError(f"510300分红缺少字段：{sorted(missing)}")
    data = dividends.loc[dividends["symbol"].astype(str).eq("510300.SH")].copy()
    for column in ("ex_date", "payment_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    data["cash_dividend_per_share"] = pd.to_numeric(
        data["cash_dividend_per_share"], errors="coerce"
    )
    if data[["ex_date", "payment_date", "cash_dividend_per_share"]].isna().any().any():
        raise ValueError("510300分红存在空日期或金额")
    if data["cash_dividend_per_share"].le(0).any():
        raise ValueError("510300分红存在非正金额")
    if not coverage.get("complete_history_confirmed"):
        raise ValueError("510300分红完整历史尚未确认")
    return data.sort_values("ex_date").reset_index(drop=True)


def _etf_horizon_return(
    holding: pd.DataFrame,
    dividend_by_date: dict[pd.Timestamp, float],
) -> dict[str, Any]:
    if holding.empty:
        raise ValueError("持有窗口为空")
    entry = holding.iloc[0]
    wealth = float(entry["etf_close"] / entry["etf_open"])
    previous_close = float(entry["etf_close"])
    dividend_count = 0
    dividend_cash_sum = 0.0
    for row in holding.iloc[1:].itertuples(index=False):
        dividend = float(dividend_by_date.get(pd.Timestamp(row.date), 0.0))
        if dividend > 0:
            dividend_count += 1
            dividend_cash_sum += dividend
        wealth *= (float(row.etf_close) + dividend) / previous_close
        previous_close = float(row.etf_close)
    exit_close = float(holding.iloc[-1]["etf_close"])
    price_return = exit_close / float(entry["etf_open"]) - 1.0
    total_return = wealth - 1.0
    return {
        "etf_price_return": price_return,
        "etf_total_return": total_return,
        "etf_dividend_return_contribution": total_return - price_return,
        "dividend_event_count": dividend_count,
        "cash_dividend_per_share_sum": dividend_cash_sum,
        "entry_day_dividend_excluded": float(
            dividend_by_date.get(pd.Timestamp(entry["date"]), 0.0)
        ),
    }


def build_forward_labels(
    signals: pd.DataFrame,
    etf: pd.DataFrame,
    price_index: pd.DataFrame,
    total_return_index: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_coverage: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_signal = {
        "date",
        "earliest_execution_date",
        "percentile_ready",
        "historical_evidence_label",
    }
    if missing := required_signal.difference(signals.columns):
        raise ValueError(f"冻结信号表缺少字段：{sorted(missing)}")
    signal_rows = signals.loc[signals["percentile_ready"].fillna(False)].copy()
    signal_rows["date"] = pd.to_datetime(signal_rows["date"], errors="coerce").dt.normalize()
    signal_rows["earliest_execution_date"] = pd.to_datetime(
        signal_rows["earliest_execution_date"], errors="coerce"
    ).dt.normalize()
    expected_signals = int(config["label_definition"]["expected_signal_count"])
    if len(signal_rows) != expected_signals:
        raise ValueError(f"有效信号应为{expected_signals}个，实际{len(signal_rows)}")
    market, market_audit = build_common_market(etf, price_index, total_return_index, config)
    distribution = _normalize_dividends(dividends, dividend_coverage)
    market_dates = set(market["date"])
    missing_ex_dates = sorted(
        set(
            distribution.loc[
                distribution["ex_date"].between(market["date"].min(), market["date"].max()),
                "ex_date",
            ]
        )
        - market_dates
    )
    missing_payment_dates = sorted(
        set(
            distribution.loc[
                distribution["payment_date"].between(
                    market["date"].min(), market["date"].max()
                ),
                "payment_date",
            ]
        )
        - market_dates
    )
    if missing_ex_dates or missing_payment_dates:
        raise ValueError("评价窗口内分红除息日或支付日不在共同交易日历")
    dividend_by_date = (
        distribution.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    )
    positions = pd.Series(np.arange(len(market), dtype=int), index=market["date"])
    horizons = [int(value) for value in config["label_definition"]["horizons_trading_days"]]
    rows: list[dict[str, Any]] = []
    for signal in signal_rows.sort_values("date").itertuples(index=False):
        observation_date = pd.Timestamp(signal.date)
        entry_date = pd.Timestamp(signal.earliest_execution_date)
        after_signal = market.loc[market["date"].gt(observation_date), "date"]
        expected_entry = pd.Timestamp(after_signal.iloc[0]) if not after_signal.empty else pd.NaT
        if pd.isna(expected_entry) or entry_date != expected_entry:
            raise ValueError(
                f"{observation_date.date()}冻结最早执行日{entry_date}与共同日历{expected_entry}不一致"
            )
        entry_index = int(positions.loc[entry_date])
        entry_row = market.iloc[entry_index]
        for horizon in horizons:
            maturity_index = entry_index + horizon - 1
            matured = maturity_index < len(market)
            observed_days = min(horizon, len(market) - entry_index)
            base: dict[str, Any] = {
                "signal_observation_date": observation_date,
                "horizon_trading_days": horizon,
                "entry_date": entry_date,
                "entry_day_is_holding_day_one": True,
                "required_holding_trading_days": horizon,
                "observed_holding_trading_days": int(observed_days),
                "maturity_date": pd.Timestamp(market.iloc[maturity_index]["date"]) if matured else pd.NaT,
                "label_status": "MATURED" if matured else "RIGHT_CENSORED_DATA_CUTOFF",
                "failure_category": "PASS" if matured else "INSUFFICIENT_TRADING_DAYS_BY_FIXED_CUTOFF",
                "label_data_cutoff": pd.Timestamp(config["protocol"]["label_data_cutoff"]),
                "etf_entry_open": float(entry_row["etf_open"]),
                "h00300_synthetic_entry_open": float(entry_row["h00300_synthetic_open"]),
                "etf_exit_close": np.nan,
                "h00300_exit_close": np.nan,
                "etf_price_return": np.nan,
                "etf_total_return": np.nan,
                "etf_dividend_return_contribution": np.nan,
                "h00300_total_return": np.nan,
                "etf_minus_h00300_return": np.nan,
                "dividend_event_count": np.nan,
                "cash_dividend_per_share_sum": np.nan,
                "entry_day_dividend_excluded": np.nan,
                "historical_evidence_label": "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS",
            }
            if matured:
                holding = market.iloc[entry_index : maturity_index + 1]
                etf_result = _etf_horizon_return(holding, dividend_by_date)
                exit_row = market.iloc[maturity_index]
                h00300_return = (
                    float(exit_row["h00300_close"])
                    / float(entry_row["h00300_synthetic_open"])
                    - 1.0
                )
                base.update(etf_result)
                base.update(
                    {
                        "etf_exit_close": float(exit_row["etf_close"]),
                        "h00300_exit_close": float(exit_row["h00300_close"]),
                        "h00300_total_return": h00300_return,
                        "etf_minus_h00300_return": etf_result["etf_total_return"]
                        - h00300_return,
                    }
                )
            rows.append(base)
    labels = pd.DataFrame(rows).sort_values(
        ["signal_observation_date", "horizon_trading_days"]
    ).reset_index(drop=True)
    leaked = SIGNAL_VALUE_COLUMNS_FORBIDDEN_IN_LABELS.intersection(labels.columns)
    if leaked:
        raise ValueError(f"标签表泄漏信号值字段：{sorted(leaked)}")
    maturity = {
        str(horizon): {
            "matured_count": int(
                labels.loc[
                    labels["horizon_trading_days"].eq(horizon), "label_status"
                ].eq("MATURED").sum()
            ),
            "right_censored_count": int(
                labels.loc[
                    labels["horizon_trading_days"].eq(horizon), "label_status"
                ].eq("RIGHT_CENSORED_DATA_CUTOFF").sum()
            ),
        }
        for horizon in horizons
    }
    expected_matured = {
        str(key): int(value)
        for key, value in config["label_definition"]["expected_matured_counts"].items()
    }
    expected_censored = {
        str(key): int(value)
        for key, value in config["label_definition"]["expected_right_censored_counts"].items()
    }
    maturity_pass = all(
        maturity[str(horizon)]["matured_count"] == expected_matured[str(horizon)]
        and maturity[str(horizon)]["right_censored_count"] == expected_censored[str(horizon)]
        for horizon in horizons
    )
    tracking_diagnostics: dict[str, Any] = {}
    for horizon in horizons:
        matured_frame = labels.loc[
            labels["horizon_trading_days"].eq(horizon)
            & labels["label_status"].eq("MATURED")
        ]
        difference = matured_frame["etf_minus_h00300_return"]
        tracking_diagnostics[str(horizon)] = {
            "observations": int(len(matured_frame)),
            "mean_etf_minus_h00300_return": _safe_number(difference.mean()),
            "median_etf_minus_h00300_return": _safe_number(difference.median()),
            "maximum_absolute_etf_minus_h00300_return": _safe_number(
                difference.abs().max()
            ),
            "pearson_correlation": _safe_number(
                matured_frame[["etf_total_return", "h00300_total_return"]]
                .corr(method="pearson")
                .iloc[0, 1]
            ),
        }
    audit = {
        "market": market_audit,
        "dividends": {
            "complete_history_confirmed": bool(dividend_coverage.get("complete_history_confirmed")),
            "event_count_all_history": int(len(distribution)),
            "event_count_through_cutoff": int(
                distribution["ex_date"].le(pd.Timestamp(config["protocol"]["label_data_cutoff"])).sum()
            ),
            "event_count_in_common_market_window": int(
                distribution["ex_date"].between(
                    market["date"].min(), market["date"].max()
                ).sum()
            ),
            "missing_ex_dates_on_common_calendar": [str(date.date()) for date in missing_ex_dates],
            "missing_payment_dates_on_common_calendar": [
                str(date.date()) for date in missing_payment_dates
            ],
        },
        "signal_count": int(len(signal_rows)),
        "label_row_count": int(len(labels)),
        "maturity_by_horizon": maturity,
        "expected_matured_counts": expected_matured,
        "expected_right_censored_counts": expected_censored,
        "maturity_counts_match_freeze": maturity_pass,
        "etf_vs_h00300_tracking_diagnostics": tracking_diagnostics,
        "label_signal_value_leakage_columns": sorted(leaked),
        "status": "PASS" if market_audit["status"] == "PASS" and maturity_pass else "BLOCKED",
    }
    return labels, audit


def frozen_bucket(percentile: float) -> str:
    if percentile < 0.20:
        return "B1_EXPENSIVE"
    if percentile < 0.40:
        return "B2"
    if percentile < 0.60:
        return "B3"
    return "B4_CHEAP"


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def hac_rank_slope(
    predictor: pd.Series,
    outcome: pd.Series,
    max_lags: int,
) -> dict[str, Any]:
    frame = pd.DataFrame({"x": predictor, "y": outcome}).dropna().reset_index(drop=True)
    n = len(frame)
    if n <= max_lags + 3:
        return {"slope": None, "standard_error": None, "t_statistic": None, "one_sided_p_value": None}
    x_rank = frame["x"].rank(method="average").to_numpy(dtype=float)
    y_rank = frame["y"].rank(method="average").to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), x_rank])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y_rank
    residual = y_rank - design @ beta
    meat = np.zeros((2, 2), dtype=float)
    for index in range(n):
        vector = design[index][:, None]
        meat += residual[index] ** 2 * (vector @ vector.T)
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        gamma = np.zeros((2, 2), dtype=float)
        for index in range(lag, n):
            current = design[index][:, None]
            previous = design[index - lag][:, None]
            gamma += residual[index] * residual[index - lag] * (current @ previous.T)
        meat += weight * (gamma + gamma.T)
    covariance = bread @ meat @ bread
    covariance *= n / (n - design.shape[1])
    variance = max(float(covariance[1, 1]), 0.0)
    standard_error = math.sqrt(variance)
    t_statistic = float(beta[1] / standard_error) if standard_error > 0 else np.nan
    p_value = 0.5 * math.erfc(t_statistic / math.sqrt(2.0)) if math.isfinite(t_statistic) else np.nan
    return {
        "slope": _safe_number(beta[1]),
        "standard_error": _safe_number(standard_error),
        "t_statistic": _safe_number(t_statistic),
        "one_sided_p_value": _safe_number(p_value),
        "max_lags": int(max_lags),
    }


def _spearman(frame: pd.DataFrame, predictor: str, outcome: str) -> float | None:
    valid = frame[[predictor, outcome]].dropna()
    if len(valid) < 2:
        return None
    return _safe_number(valid.corr(method="spearman").iloc[0, 1])


def compute_target_diagnostic(
    joined: pd.DataFrame,
    outcome: str,
    horizon: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = joined.loc[
        joined["horizon_trading_days"].eq(horizon)
        & joined["label_status"].eq("MATURED"),
        ["signal_observation_date", "raw_ey_percentile_60m", outcome],
    ].dropna().sort_values("signal_observation_date").reset_index(drop=True)
    frame["bucket"] = frame["raw_ey_percentile_60m"].map(frozen_bucket)
    bucket_order = ["B1_EXPENSIVE", "B2", "B3", "B4_CHEAP"]
    buckets: list[dict[str, Any]] = []
    for bucket in bucket_order:
        values = frame.loc[frame["bucket"].eq(bucket), outcome]
        buckets.append(
            {
                "bucket": bucket,
                "observations": int(len(values)),
                "mean_return": _safe_number(values.mean()),
                "median_return": _safe_number(values.median()),
                "positive_return_ratio": _safe_number(values.gt(0).mean()) if len(values) else None,
            }
        )
    means = [row["mean_return"] for row in buckets]
    all_means_available = all(value is not None for value in means)
    monotonic = bool(
        all_means_available
        and all(float(means[index + 1]) >= float(means[index]) for index in range(3))
    )
    mean_spread = (
        float(means[-1]) - float(means[0]) if all_means_available else None
    )
    medians = [row["median_return"] for row in buckets]
    median_spread = (
        float(medians[-1]) - float(medians[0])
        if medians[0] is not None and medians[-1] is not None
        else None
    )
    midpoint = len(frame) // 2
    halves = {
        "first": {
            "observations": int(midpoint),
            "spearman_ic": _spearman(
                frame.iloc[:midpoint], "raw_ey_percentile_60m", outcome
            ),
        },
        "second": {
            "observations": int(len(frame) - midpoint),
            "spearman_ic": _spearman(
                frame.iloc[midpoint:], "raw_ey_percentile_60m", outcome
            ),
        },
    }
    lag_map = config["diagnostic_definition"]["hac_monthly_max_lags"]
    hac = hac_rank_slope(
        frame["raw_ey_percentile_60m"], frame[outcome], int(lag_map[horizon])
    )
    return {
        "outcome": outcome,
        "horizon_trading_days": horizon,
        "matured_observations": int(len(frame)),
        "first_observation_date": str(frame["signal_observation_date"].min().date()) if len(frame) else None,
        "last_observation_date": str(frame["signal_observation_date"].max().date()) if len(frame) else None,
        "spearman_ic": _spearman(frame, "raw_ey_percentile_60m", outcome),
        "hac_rank_regression": hac,
        "chronological_halves": halves,
        "buckets": buckets,
        "bucket_mean_returns_nondecreasing": monotonic,
        "cheapest_minus_expensive_mean_return": _safe_number(mean_spread),
        "cheapest_minus_expensive_median_return": _safe_number(median_spread),
    }


def build_predictive_report(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    label_audit: dict[str, Any],
    hash_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    signal_values = signals.loc[
        signals["percentile_ready"].fillna(False),
        ["date", "raw_ey_percentile_60m"],
    ].rename(columns={"date": "signal_observation_date"})
    joined = labels.merge(signal_values, on="signal_observation_date", validate="many_to_one")
    diagnostics: dict[str, Any] = {}
    for horizon in config["label_definition"]["horizons_trading_days"]:
        horizon_int = int(horizon)
        diagnostics[str(horizon_int)] = {
            "etf_total_return": compute_target_diagnostic(
                joined, "etf_total_return", horizon_int, config
            ),
            "h00300_total_return": compute_target_diagnostic(
                joined, "h00300_total_return", horizon_int, config
            ),
        }
    primary_horizon = str(config["label_definition"]["primary_horizon_trading_days"])
    primary = diagnostics[primary_horizon]["etf_total_return"]
    confirm = diagnostics[primary_horizon]["h00300_total_return"]
    gate = config["primary_predictive_gate"]
    bucket_counts = [row["observations"] for row in primary["buckets"]]
    first_ic = primary["chronological_halves"]["first"]["spearman_ic"]
    second_ic = primary["chronological_halves"]["second"]["spearman_ic"]
    p_value = primary["hac_rank_regression"]["one_sided_p_value"]
    checks = {
        "minimum_matured_observations": primary["matured_observations"]
        >= int(gate["minimum_matured_observations"]),
        "minimum_observations_per_bucket": min(bucket_counts)
        >= int(gate["minimum_observations_per_bucket"]),
        "etf_spearman_ic_strictly_positive": primary["spearman_ic"] is not None
        and primary["spearman_ic"] > 0,
        "etf_hac_one_sided_p_value_at_most_0_10": p_value is not None
        and p_value <= float(gate["hac_one_sided_p_value_maximum"]),
        "both_chronological_half_ics_strictly_positive": first_ic is not None
        and second_ic is not None
        and first_ic > 0
        and second_ic > 0,
        "etf_bucket_mean_returns_nondecreasing": primary[
            "bucket_mean_returns_nondecreasing"
        ],
        "etf_cheapest_minus_expensive_mean_positive": primary[
            "cheapest_minus_expensive_mean_return"
        ]
        is not None
        and primary["cheapest_minus_expensive_mean_return"] > 0,
        "etf_cheapest_minus_expensive_median_positive": primary[
            "cheapest_minus_expensive_median_return"
        ]
        is not None
        and primary["cheapest_minus_expensive_median_return"] > 0,
        "h00300_spearman_confirmation_positive": confirm["spearman_ic"] is not None
        and confirm["spearman_ic"] > 0,
        "h00300_cheapest_minus_expensive_mean_positive": confirm[
            "cheapest_minus_expensive_mean_return"
        ]
        is not None
        and confirm["cheapest_minus_expensive_mean_return"] > 0,
    }
    passed = all(checks.values())
    status = gate["pass_status"] if passed else gate["fail_status"]
    failed_checks = [name for name, value in checks.items() if not value]
    stop_reason = (
        "PRIMARY_BUCKET_SUPPORT_INSUFFICIENT"
        if failed_checks == ["minimum_observations_per_bucket"]
        else "PRIMARY_PREDICTIVE_GATE_FAILED"
    )
    governance = {
        "forward_labels_generated": True,
        "predictive_ic_calculated": True,
        "historical_strategy_return_calculated": False,
        "historical_position_mapping_performed": False,
        "current_position_mapping_performed": False,
        "target_shares_generated": False,
        "orders_generated": False,
        "broker_connection_performed": False,
        "alpha_pass": False,
        "upstream_model_files_mutated": False,
    }
    return {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(ZoneInfo(config["protocol"]["timezone"])).isoformat(),
        "status": status,
        "historical_evidence_label": "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS",
        "hash_audit": hash_audit,
        "label_audit": label_audit,
        "diagnostics": diagnostics,
        "primary_predictive_gate": {
            "horizon_trading_days": int(primary_horizon),
            "target": "etf_total_return",
            "checks": checks,
            "passed": passed,
            "failed_checks": failed_checks,
            "stop_reason": None if passed else stop_reason,
        },
        "trial_and_multiplicity": config["trial_and_multiplicity"],
        "true_forward": config["true_forward"],
        "governance": governance,
        "artifacts": config["artifacts"],
    }


def _format_number(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _format_percent(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.2%}"


def _format_probability(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4g}"


def render_markdown(report: dict[str, Any]) -> str:
    maturity = report["label_audit"]["maturity_by_horizon"]
    lines = [
        "# VAL01_RAW_EY_5Y_V1 未来收益预测诊断",
        "",
        f"> 状态：`{report['status']}`。全部结果均为受污染历史诊断，不是严格样本外Alpha。",
        "",
        "## 标签与成熟度审计",
        "",
        f"- 标签行数：{report['label_audit']['label_row_count']}；有效信号：{report['label_audit']['signal_count']}。",
    ]
    for horizon in (60, 120, 242):
        item = maturity[str(horizon)]
        lines.append(
            f"- {horizon}日：成熟{item['matured_count']}，右删失{item['right_censored_count']}。"
        )
    market = report["label_audit"]["market"]
    lines.extend(
        [
            f"- 共同日历截至冻结截止日：{market['common_calendar_rows_through_cutoff']}日；H00300独有日期：{market['h00300_only_dates']}。",
            f"- 合成H00300收盘最大误差：{market['synthetic_total_return_close_maximum_error']:.3e}。",
            "- 标签表不包含EY、分位、仓位或订单字段。",
            "",
            "## IC与分档诊断",
            "",
            "|期限|目标|样本|Spearman IC|HAC t|单侧p|均值单调|最高档-最低档均值|",
            "|---:|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for horizon in (60, 120, 242):
        for target in ("etf_total_return", "h00300_total_return"):
            row = report["diagnostics"][str(horizon)][target]
            lines.append(
                "|{}|{}|{}|{}|{}|{}|{}|{}|".format(
                    horizon,
                    target,
                    row["matured_observations"],
                    _format_number(row["spearman_ic"]),
                    _format_number(row["hac_rank_regression"]["t_statistic"]),
                    _format_probability(row["hac_rank_regression"]["one_sided_p_value"]),
                    "是" if row["bucket_mean_returns_nondecreasing"] else "否",
                    _format_percent(row["cheapest_minus_expensive_mean_return"]),
                )
            )
    gate = report["primary_predictive_gate"]
    primary_etf = report["diagnostics"]["242"]["etf_total_return"]
    primary_h00300 = report["diagnostics"]["242"]["h00300_total_return"]
    lines.extend(
        [
            "",
            "## 242日冻结分档明细",
            "",
            "|分档|样本|510300均值|510300中位数|正收益率|H00300均值|",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for etf_bucket, index_bucket in zip(
        primary_etf["buckets"], primary_h00300["buckets"], strict=True
    ):
        lines.append(
            "|{}|{}|{}|{}|{}|{}|".format(
                etf_bucket["bucket"],
                etf_bucket["observations"],
                _format_percent(etf_bucket["mean_return"]),
                _format_percent(etf_bucket["median_return"]),
                _format_percent(etf_bucket["positive_return_ratio"]),
                _format_percent(index_bucket["mean_return"]),
            )
        )
    lines.extend(
        [
            "",
            f"- 510300前/后半段IC：{_format_number(primary_etf['chronological_halves']['first']['spearman_ic'])}/{_format_number(primary_etf['chronological_halves']['second']['spearman_ic'])}。",
            f"- 242日510300与H00300标签Pearson相关：{_format_number(report['label_audit']['etf_vs_h00300_tracking_diagnostics']['242']['pearson_correlation'])}。",
            "",
        ]
    )
    lines.extend(["", "## 242日主要硬门槛", ""])
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`：`{'PASS' if passed else 'FAIL'}`。")
    lines.extend(
        [
            "",
            f"主要屏幕结论：`{report['status']}`。",
            f"停止原因：`{gate['stop_reason'] or 'NONE'}`。",
            "",
            "## 治理边界",
            "",
            "- 没有生成历史或当前仓位、目标份额、订单或券商动作。",
            "- 没有运行策略收益、成本压力、稳定性或组合回测。",
            "- `alpha_pass=false`；若主要屏幕失败，必须停止该候选的策略回测。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    labels: pd.DataFrame,
    report: dict[str, Any],
    config: dict[str, Any],
) -> None:
    artifacts = config["artifacts"]
    label_path = ROOT / artifacts["label_table"]
    report_path = ROOT / artifacts["report_json"]
    markdown_path = ROOT / artifacts["report_markdown"]
    for path in (label_path, report_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_label = label_path.with_suffix(label_path.suffix + ".tmp")
    labels.to_parquet(temporary_label, index=False)
    temporary_label.replace(label_path)
    report["artifacts"]["label_table_sha256"] = sha256_file(label_path)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_report.replace(report_path)
    temporary_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary_markdown.write_text(render_markdown(report), encoding="utf-8")
    temporary_markdown.replace(markdown_path)
