"""510300宏观压力规避V1：点时状态、事件门与条件式组合回测。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_macro_stress_avoidance_v1.yaml"
TIME_ZONE = ZoneInfo("Asia/Shanghai")
TRADING_DAYS_PER_YEAR = 242


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取冻结协议。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload["protocol"]["study_id"] != "510300_MACRO_STRESS_AVOIDANCE_V1":
        raise ValueError("研究编号不匹配")
    return payload


def _calendar_trailing_quantile(
    dates: pd.Series,
    values: pd.Series,
    *,
    years: int,
    quantile: float,
) -> tuple[pd.Series, pd.Series]:
    """计算严格排除当期、覆盖完整日历年的历史分位数。"""

    timestamps = pd.to_datetime(dates).reset_index(drop=True)
    numeric = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    thresholds = pd.Series(np.nan, index=numeric.index, dtype="float64")
    eligible = pd.Series(False, index=numeric.index, dtype="bool")
    valid = numeric.notna()
    if not valid.any():
        return thresholds, eligible
    first_valid_date = timestamps.loc[valid].iloc[0]
    first_eligible_date = first_valid_date + pd.DateOffset(years=years)
    for position in numeric.index[valid]:
        current_date = timestamps.iloc[position]
        if current_date < first_eligible_date:
            continue
        lower_bound = current_date - pd.DateOffset(years=years)
        history_mask = (
            valid
            & timestamps.ge(lower_bound)
            & timestamps.lt(current_date)
        )
        history = numeric.loc[history_mask].to_numpy(dtype="float64")
        if history.size == 0:
            continue
        thresholds.iloc[position] = float(
            np.quantile(history, quantile, method="linear")
        )
        eligible.iloc[position] = True
    return thresholds, eligible


def _prepare_fdr_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """构造FDR007流动性缺口及严格滚动阈值。"""

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True).dt.tz_convert(
        TIME_ZONE
    )
    result = result.sort_values("date").reset_index(drop=True)
    value = pd.to_numeric(result["first_release_value"], errors="raise")
    short_window = int(config["factor_rules"]["liquidity"]["median_short_observations"])
    long_window = int(config["factor_rules"]["liquidity"]["median_long_observations"])
    result["fdr007"] = value
    result["fdr_median_5"] = value.rolling(short_window, min_periods=short_window).median()
    result["fdr_median_60"] = value.rolling(long_window, min_periods=long_window).median()
    result["liquidity_gap"] = result["fdr_median_5"] - result["fdr_median_60"]
    percentile = float(config["protocol"]["percentile"])
    for years in (3, 5):
        threshold, eligible = _calendar_trailing_quantile(
            result["date"],
            result["liquidity_gap"],
            years=years,
            quantile=percentile,
        )
        result[f"liquidity_q90_{years}y"] = threshold
        result[f"liquidity_history_eligible_{years}y"] = eligible
        stress = pd.Series(pd.NA, index=result.index, dtype="Int64")
        stress.loc[eligible] = (
            result.loc[eligible, "liquidity_gap"]
            > result.loc[eligible, f"liquidity_q90_{years}y"]
        ).astype(int)
        result[f"liquidity_stress_{years}y"] = stress
    return result


def _prepare_fx_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """构造USD/CNY二十观察日对数变化及严格滚动阈值。"""

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True).dt.tz_convert(
        TIME_ZONE
    )
    result = result.sort_values("date").reset_index(drop=True)
    value = pd.to_numeric(result["first_release_value"], errors="raise")
    lag = int(config["factor_rules"]["fx"]["lag_observations"])
    result["usdcny_midpoint"] = value
    result["fx20"] = np.log(value / value.shift(lag))
    percentile = float(config["protocol"]["percentile"])
    for years in (3, 5):
        threshold, eligible = _calendar_trailing_quantile(
            result["date"],
            result["fx20"],
            years=years,
            quantile=percentile,
        )
        result[f"fx_q90_{years}y"] = threshold
        result[f"fx_history_eligible_{years}y"] = eligible
        stress = pd.Series(pd.NA, index=result.index, dtype="Int64")
        stress.loc[eligible] = (
            result.loc[eligible, "fx20"] > result.loc[eligible, f"fx_q90_{years}y"]
        ).astype(int)
        result[f"fx_stress_{years}y"] = stress
    return result


def _prepare_monthly_stress(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """按参考月首发值构造PMI或社融t-3压力状态。"""

    if kind not in {"growth", "credit"}:
        raise ValueError(f"未知月频压力类型：{kind}")
    result = frame.copy().sort_values("reference_period").reset_index(drop=True)
    result["reference_period"] = pd.PeriodIndex(result["reference_period"], freq="M")
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True).dt.tz_convert(
        TIME_ZONE
    )
    result["value"] = pd.to_numeric(result["first_release_value"], errors="raise")
    value_by_period = dict(zip(result["reference_period"], result["value"], strict=True))
    result["value_t_minus_3"] = [
        value_by_period.get(period - 3, np.nan) for period in result["reference_period"]
    ]
    if kind == "growth":
        result["stress"] = (
            result["value"].lt(50.0)
            & result["value"].lt(result["value_t_minus_3"])
        ).where(result["value_t_minus_3"].notna())
    else:
        result["stress"] = result["value"].lt(result["value_t_minus_3"]).where(
            result["value_t_minus_3"].notna()
        )
    result["stress"] = result["stress"].astype("Int64")
    result["reference_period"] = result["reference_period"].astype(str)
    return result


def _asof_join(
    panel: pd.DataFrame,
    source: pd.DataFrame,
    *,
    prefix: str,
    source_columns: Iterable[str],
) -> pd.DataFrame:
    """仅把在收盘信号截止时点已经公开的数据拼入面板。"""

    columns = ["available_at", *source_columns]
    right = source.loc[:, columns].copy()
    right = right.rename(columns={column: f"{prefix}_{column}" for column in columns})
    return pd.merge_asof(
        panel.sort_values("signal_cutoff"),
        right.sort_values(f"{prefix}_available_at"),
        left_on="signal_cutoff",
        right_on=f"{prefix}_available_at",
        direction="backward",
        allow_exact_matches=True,
    )


def apply_recovery_hysteresis(
    raw_targets: pd.Series,
    *,
    confirmation_days: int,
    upward_step: float,
    initial_target: float,
) -> tuple[pd.Series, pd.Series]:
    """风险恶化立即执行，改善同档确认后每次仅恢复一档。"""

    output = pd.Series(np.nan, index=raw_targets.index, dtype="float64")
    counters = pd.Series(0, index=raw_targets.index, dtype="int64")
    current = float(initial_target)
    candidate_band: float | None = None
    confirmation_count = 0
    for index, raw_value in raw_targets.items():
        if pd.isna(raw_value):
            candidate_band = None
            confirmation_count = 0
            continue
        raw = float(raw_value)
        if raw < current - 1e-12:
            current = raw
            candidate_band = None
            confirmation_count = 0
        elif raw > current + 1e-12:
            if candidate_band is None or not math.isclose(candidate_band, raw):
                candidate_band = raw
                confirmation_count = 1
            else:
                confirmation_count += 1
            if confirmation_count >= confirmation_days:
                current = min(current + upward_step, raw)
                candidate_band = None
                confirmation_count = 0
        else:
            candidate_band = None
            confirmation_count = 0
        output.loc[index] = current
        counters.loc[index] = confirmation_count
    return output, counters


def build_point_in_time_panel(
    market: pd.DataFrame,
    fdr: pd.DataFrame,
    fx: pd.DataFrame,
    pmi: pd.DataFrame,
    tsf: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """生成仅使用各交易日15:00前已知首发值的宏观状态面板。"""

    protocol = config["protocol"]
    prices = market.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.loc[
        prices["date"].between(protocol["data_start"], protocol["data_end"])
    ].sort_values("date").reset_index(drop=True)
    if prices.empty or prices["date"].duplicated().any():
        raise ValueError("510300行情在冻结区间为空或日期重复")
    cutoff_naive = prices["date"] + pd.Timedelta(
        config["point_in_time"]["signal_cutoff"]
    )
    prices["signal_cutoff"] = cutoff_naive.dt.tz_localize(TIME_ZONE)

    fdr_features = _prepare_fdr_features(fdr, config)
    fx_features = _prepare_fx_features(fx, config)
    pmi_features = _prepare_monthly_stress(pmi, kind="growth")
    tsf_features = _prepare_monthly_stress(tsf, kind="credit")

    fdr_columns = [
        "date",
        "fdr007",
        "fdr_median_5",
        "fdr_median_60",
        "liquidity_gap",
        "liquidity_q90_3y",
        "liquidity_q90_5y",
        "liquidity_history_eligible_3y",
        "liquidity_history_eligible_5y",
        "liquidity_stress_3y",
        "liquidity_stress_5y",
    ]
    fx_columns = [
        "date",
        "usdcny_midpoint",
        "fx20",
        "fx_q90_3y",
        "fx_q90_5y",
        "fx_history_eligible_3y",
        "fx_history_eligible_5y",
        "fx_stress_3y",
        "fx_stress_5y",
    ]
    monthly_columns = ["reference_period", "value", "value_t_minus_3", "stress"]
    panel = _asof_join(
        prices,
        fdr_features,
        prefix="fdr",
        source_columns=fdr_columns,
    )
    panel = _asof_join(
        panel,
        fx_features,
        prefix="fx",
        source_columns=fx_columns,
    )
    panel = _asof_join(
        panel,
        pmi_features,
        prefix="pmi",
        source_columns=monthly_columns,
    )
    panel = _asof_join(
        panel,
        tsf_features,
        prefix="tsf",
        source_columns=monthly_columns,
    )

    for years in (3, 5):
        eligible = (
            panel[f"fdr_liquidity_history_eligible_{years}y"].fillna(False).astype(bool)
            & panel[f"fx_fx_history_eligible_{years}y"].fillna(False).astype(bool)
            & panel["pmi_stress"].notna()
            & panel["tsf_stress"].notna()
        )
        panel[f"model_eligible_{years}y"] = eligible
        score = pd.Series(pd.NA, index=panel.index, dtype="Int64")
        score.loc[eligible] = (
            panel.loc[eligible, f"fdr_liquidity_stress_{years}y"].astype(int)
            + panel.loc[eligible, f"fx_fx_stress_{years}y"].astype(int)
            + panel.loc[eligible, "pmi_stress"].astype(int)
            + panel.loc[eligible, "tsf_stress"].astype(int)
        )
        panel[f"stress_score_{years}y"] = score
        raw_target = score.map({0: 1.0, 1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0})
        panel[f"raw_target_{years}y"] = raw_target.astype("float64")
        target, counter = apply_recovery_hysteresis(
            panel[f"raw_target_{years}y"],
            confirmation_days=int(
                config["recovery_rule"]["improving_confirmation_trading_days"]
            ),
            upward_step=float(config["recovery_rule"]["upward_step"]),
            initial_target=float(config["recovery_rule"]["initial_target_position"]),
        )
        panel[f"target_position_{years}y"] = target
        panel[f"recovery_counter_{years}y"] = counter

    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("点时面板日期重复或未递增")
    for prefix in ("fdr", "fx", "pmi", "tsf"):
        available = panel[f"{prefix}_available_at"]
        if (available.notna() & available.gt(panel["signal_cutoff"])).any():
            raise AssertionError(f"{prefix}发生未来数据泄漏")
    return panel


def _attach_forward_outcomes(
    panel: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """计算从事件日收盘到未来第h个交易日收盘的含分红总收益。"""

    result = panel.copy().sort_values("date").reset_index(drop=True)
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"]).dt.normalize()
    dividend_by_date = events.groupby("ex_date")["cash_dividend_per_share"].sum()
    result["cash_dividend_per_share"] = result["date"].map(dividend_by_date).fillna(0.0)
    previous_close = result["close"].shift(1)
    result["etf_total_return_1d"] = (
        (result["close"] + result["cash_dividend_per_share"]) / previous_close - 1.0
    )

    index_frame = benchmark.loc[:, ["date", "close"]].copy()
    index_frame["date"] = pd.to_datetime(index_frame["date"]).dt.normalize()
    index_frame = index_frame.rename(columns={"close": "h00300_close"})
    result = result.merge(index_frame, on="date", how="left", validate="one_to_one")
    if result["h00300_close"].isna().any():
        missing = result.loc[result["h00300_close"].isna(), "date"].dt.date.tolist()
        raise ValueError(f"H00300在510300交易日缺值：{missing[:10]}")
    result["h00300_total_return_1d"] = result["h00300_close"].pct_change()

    etf_factors = (1.0 + result["etf_total_return_1d"].fillna(0.0)).cumprod().to_numpy()
    benchmark_factors = (
        1.0 + result["h00300_total_return_1d"].fillna(0.0)
    ).cumprod().to_numpy()
    positions = np.arange(len(result))
    for horizon in horizons:
        valid = positions + int(horizon) < len(result)
        etf_forward = np.full(len(result), np.nan, dtype="float64")
        benchmark_forward = np.full(len(result), np.nan, dtype="float64")
        etf_forward[valid] = (
            etf_factors[positions[valid] + int(horizon)] / etf_factors[positions[valid]] - 1.0
        )
        benchmark_forward[valid] = (
            benchmark_factors[positions[valid] + int(horizon)]
            / benchmark_factors[positions[valid]]
            - 1.0
        )
        result[f"etf_forward_{horizon}d"] = etf_forward
        result[f"h00300_forward_{horizon}d"] = benchmark_forward
        result[f"relative_forward_{horizon}d"] = etf_forward - benchmark_forward
    return result


def _finite_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "median": None, "negative_fraction": None}
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "negative_fraction": float(numeric.lt(0.0).mean()),
    }


def evaluate_event_gate(
    panel: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """运行唯一5年模型事件研究并给出是否允许组合回测。"""

    event_config = config["event_study"]
    outcome_panel = _attach_forward_outcomes(
        panel,
        dividends,
        benchmark,
        event_config["horizons_trading_days"],
    )
    previous_eligible = outcome_panel["model_eligible_5y"].shift(1).fillna(False)
    previous_score = outcome_panel["stress_score_5y"].shift(1)
    raw_onset = (
        outcome_panel["model_eligible_5y"]
        & previous_eligible
        & previous_score.lt(2)
        & outcome_panel["stress_score_5y"].ge(2)
    )
    onset_positions = np.flatnonzero(raw_onset.to_numpy())
    kept_positions: list[int] = []
    spacing = int(event_config["minimum_spacing_trading_days"])
    for position in onset_positions:
        if not kept_positions or position - kept_positions[-1] >= spacing:
            kept_positions.append(int(position))
    events = outcome_panel.iloc[kept_positions].copy()
    events.insert(0, "event_id", [f"EVENT_{number:03d}" for number in range(1, len(events) + 1)])
    primary = int(event_config["primary_horizon_trading_days"])
    primary_column = f"etf_forward_{primary}d"
    events["primary_complete"] = events[primary_column].notna()
    complete_events = events.loc[events["primary_complete"]].copy()

    controls = outcome_panel.loc[
        outcome_panel["model_eligible_5y"]
        & outcome_panel["stress_score_5y"].lt(2)
        & outcome_panel[primary_column].notna()
    ].copy()
    event_primary = _finite_summary(complete_events[primary_column])
    control_primary = _finite_summary(controls[primary_column])
    event_secondary = _finite_summary(complete_events[f"relative_forward_{primary}d"])
    control_secondary = _finite_summary(controls[f"relative_forward_{primary}d"])

    minimum_events = int(event_config["minimum_independent_events"])
    count_pass = len(complete_events) >= minimum_events
    mean_pass = bool(
        event_primary["mean"] is not None
        and control_primary["mean"] is not None
        and event_primary["mean"] < control_primary["mean"]
    )
    median_pass = bool(
        event_primary["median"] is not None
        and control_primary["median"] is not None
        and event_primary["median"] < control_primary["median"]
    )
    negative_pass = bool(
        event_primary["negative_fraction"] is not None
        and event_primary["negative_fraction"]
        >= float(event_config["minimum_negative_fraction"])
    )
    distinct_years = sorted(complete_events["date"].dt.year.unique().tolist())
    year_pass = len(distinct_years) >= int(event_config["minimum_distinct_event_years"])

    worst_removal: dict[str, Any]
    if len(complete_events) >= 2 and control_primary["mean"] is not None:
        worst_index = complete_events[primary_column].idxmin()
        reduced = complete_events.drop(index=worst_index)
        reduced_summary = _finite_summary(reduced[primary_column])
        worst_removal_pass = bool(
            reduced_summary["mean"] is not None
            and reduced_summary["median"] is not None
            and reduced_summary["mean"] < control_primary["mean"]
            and reduced_summary["median"] < control_primary["median"]
        )
        worst_removal = {
            "removed_event_id": str(complete_events.loc[worst_index, "event_id"]),
            "removed_event_date": str(complete_events.loc[worst_index, "date"].date()),
            "removed_return": float(complete_events.loc[worst_index, primary_column]),
            "remaining": reduced_summary,
            "pass": worst_removal_pass,
        }
    else:
        worst_removal_pass = False
        worst_removal = {
            "removed_event_id": None,
            "removed_event_date": None,
            "removed_return": None,
            "remaining": _finite_summary(pd.Series(dtype="float64")),
            "pass": False,
        }

    gates = {
        "minimum_eight_independent_complete_events": count_pass,
        "event_mean_below_nonpressure": mean_pass,
        "event_median_below_nonpressure": median_pass,
        "minimum_sixty_percent_negative": negative_pass,
        "delete_worst_direction_unchanged": worst_removal_pass,
        "events_span_at_least_two_years": year_pass,
    }
    passed = all(gates.values())
    if not count_pass:
        decision = "REJECTED_FROZEN_INSUFFICIENT_EVENT_SUPPORT_NO_RESCUE"
    elif not passed:
        decision = "REJECTED_FROZEN_EVENT_MECHANISM_FAILED_NO_RESCUE"
    else:
        decision = "PASS_EVENT_GATE_PORTFOLIO_RUN_ALLOWED_ONCE"

    horizon_summaries: dict[str, Any] = {}
    for horizon in event_config["horizons_trading_days"]:
        horizon_summaries[str(horizon)] = {
            "event_etf_total_return": _finite_summary(events[f"etf_forward_{horizon}d"]),
            "control_etf_total_return": _finite_summary(
                outcome_panel.loc[
                    outcome_panel["model_eligible_5y"]
                    & outcome_panel["stress_score_5y"].lt(2),
                    f"etf_forward_{horizon}d",
                ]
            ),
            "event_relative_to_h00300": _finite_summary(
                events[f"relative_forward_{horizon}d"]
            ),
        }

    first_eligible = outcome_panel.loc[outcome_panel["model_eligible_5y"], "date"]
    report = {
        "decision": decision,
        "passed": passed,
        "first_model_eligible_date": (
            str(first_eligible.iloc[0].date()) if not first_eligible.empty else None
        ),
        "last_model_eligible_date": (
            str(first_eligible.iloc[-1].date()) if not first_eligible.empty else None
        ),
        "eligible_trading_days": int(outcome_panel["model_eligible_5y"].sum()),
        "raw_onsets": int(raw_onset.sum()),
        "independent_onsets": int(len(events)),
        "independent_complete_primary_events": int(len(complete_events)),
        "distinct_complete_event_years": distinct_years,
        "primary_horizon_trading_days": primary,
        "primary_event_summary": event_primary,
        "primary_control_summary": control_primary,
        "secondary_event_summary": event_secondary,
        "secondary_control_summary": control_secondary,
        "worst_event_removal": worst_removal,
        "horizon_summaries": horizon_summaries,
        "gates": gates,
        "failed_gates": [name for name, passed_gate in gates.items() if not passed_gate],
    }
    event_columns = [
        "event_id",
        "date",
        "stress_score_5y",
        "target_position_5y",
        "fdr_liquidity_stress_5y",
        "pmi_stress",
        "tsf_stress",
        "fx_fx_stress_5y",
        "primary_complete",
        *[
            column
            for horizon in event_config["horizons_trading_days"]
            for column in (
                f"etf_forward_{horizon}d",
                f"h00300_forward_{horizon}d",
                f"relative_forward_{horizon}d",
            )
        ],
    ]
    control_columns = [
        "date",
        "stress_score_5y",
        *[
            column
            for horizon in event_config["horizons_trading_days"]
            for column in (
                f"etf_forward_{horizon}d",
                f"h00300_forward_{horizon}d",
                f"relative_forward_{horizon}d",
            )
        ],
    ]
    return report, events.loc[:, event_columns], controls.loc[:, control_columns]


def _costs(payload: dict[str, Any], lot_size: int) -> BacktestCosts:
    return BacktestCosts(
        commission_rate=float(payload["commission_rate"]),
        minimum_commission_cny=float(payload["minimum_commission_cny"]),
        stamp_duty_rate=float(payload["stamp_duty_rate"]),
        slippage_bps=float(payload["slippage_bps_one_way"]),
        lot_size=int(lot_size),
        cash_annual_rate=float(payload["cash_annual_rate"]),
    )


def _complete_round_trips_by_year(targets: pd.DataFrame) -> dict[str, int]:
    """按首次降至不足满仓到恢复满仓定义完整往返。"""

    active = False
    counts: dict[str, int] = {}
    for row in targets.sort_values("date").itertuples(index=False):
        target = float(row.target_position)
        if target < 1.0 - 1e-12 and not active:
            active = True
        elif target >= 1.0 - 1e-12 and active:
            year = str(pd.Timestamp(row.date).year)
            counts[year] = counts.get(year, 0) + 1
            active = False
    return counts


def _index_summary(
    benchmark: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    frame = benchmark.loc[:, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[frame["date"].between(start_date, end_date)].sort_values("date")
    if len(frame) < 2:
        raise ValueError("H00300有效评估行数不足")
    returns = frame["close"].pct_change().fillna(0.0)
    elapsed_days = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days, 1)
    total_return = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    sharpe = _sharpe(returns)
    return {
        "start_date": str(frame["date"].iloc[0].date()),
        "end_date": str(frame["date"].iloc[-1].date()),
        "observations": int(len(frame)),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_zero_cash_rate": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _sharpe(returns: pd.Series) -> float | None:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    volatility = float(numeric.std(ddof=1))
    if volatility <= 0.0:
        return None
    return float(numeric.mean() / volatility * math.sqrt(TRADING_DAYS_PER_YEAR))


def _run_path(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    costs: BacktestCosts,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start_date = pd.Timestamp(targets["date"].min())
    end_date = pd.Timestamp(targets["date"].max())
    ledger, trades = run_long_cash_backtest(
        market,
        dividends,
        targets,
        initial_cash=initial_cash,
        costs=costs,
        start_date=start_date,
        end_date=end_date,
    )
    return ledger, trades, summarize_backtest(ledger, trades, initial_cash)


def _evaluate_model(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict[str, Any],
    *,
    years: int,
    cost_scenario: str,
) -> dict[str, Any]:
    eligible = panel.loc[panel[f"model_eligible_{years}y"]].copy()
    if len(eligible) < 2:
        raise ValueError(f"{years}年模型没有足够的有效交易日")
    targets = eligible.loc[:, ["date", f"target_position_{years}y"]].rename(
        columns={f"target_position_{years}y": "target_position"}
    )
    targets["signal_reason"] = f"冻结宏观压力状态{years}Y"
    initial_cash = float(config["execution"]["initial_cash_cny"])
    lot_size = int(config["execution"]["lot_size_shares"])
    scenario_costs = _costs(config["costs"][cost_scenario], lot_size)
    ledger, trades, summary = _run_path(
        market, dividends, targets, scenario_costs, initial_cash
    )

    buy_hold_targets = targets.loc[:, ["date"]].copy()
    buy_hold_targets["target_position"] = 1.0
    buy_hold_targets["signal_reason"] = "冻结510300含分红买入持有基准"
    base_costs = _costs(config["costs"]["base"], lot_size)
    buy_hold_ledger, buy_hold_trades, buy_hold_summary = _run_path(
        market, dividends, buy_hold_targets, base_costs, initial_cash
    )
    index_summary = _index_summary(
        benchmark,
        pd.Timestamp(targets["date"].min()),
        pd.Timestamp(targets["date"].max()),
    )
    round_trips = _complete_round_trips_by_year(targets)
    return {
        "years": years,
        "cost_scenario": cost_scenario,
        "targets": targets,
        "ledger": ledger,
        "trades": trades,
        "summary": summary,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
        "buy_hold_summary": buy_hold_summary,
        "h00300_summary": index_summary,
        "annualized_excess_vs_buy_hold": float(summary["cagr"] - buy_hold_summary["cagr"]),
        "annualized_excess_vs_h00300": float(summary["cagr"] - index_summary["cagr"]),
        "max_drawdown_ratio_vs_buy_hold": (
            float(abs(summary["max_drawdown"]) / abs(buy_hold_summary["max_drawdown"]))
            if buy_hold_summary["max_drawdown"] < 0
            else None
        ),
        "complete_round_trips_by_year": round_trips,
        "maximum_complete_round_trips_in_any_year": max(round_trips.values(), default=0),
    }


def _leave_one_year_out_sharpes(ledger: pd.DataFrame) -> dict[str, float | None]:
    years = sorted(ledger["date"].dt.year.unique().tolist())
    return {
        str(year): _sharpe(ledger.loc[ledger["date"].dt.year.ne(year), "daily_return"])
        for year in years
    }


def _aligned_log_excess(
    strategy_ledger: pd.DataFrame,
    buy_hold_ledger: pd.DataFrame,
) -> pd.DataFrame:
    merged = strategy_ledger.loc[:, ["date", "daily_return", "target_position"]].merge(
        buy_hold_ledger.loc[:, ["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    merged["log_excess"] = np.log1p(merged["daily_return_strategy"]) - np.log1p(
        merged["daily_return_buy_hold"]
    )
    return merged


def _single_year_excess_share(log_excess: pd.DataFrame) -> tuple[float | None, dict[str, float]]:
    contributions = log_excess.groupby(log_excess["date"].dt.year)["log_excess"].sum()
    total = float(contributions.sum())
    by_year = {str(int(year)): float(value) for year, value in contributions.items()}
    if total <= 0.0:
        return None, by_year
    maximum_positive = max((float(value) for value in contributions if value > 0.0), default=0.0)
    return float(maximum_positive / total), by_year


def _single_risk_cycle_excess_share(
    log_excess: pd.DataFrame,
) -> tuple[float | None, list[dict[str, Any]]]:
    cycles: list[dict[str, Any]] = []
    active_rows: list[int] = []
    for index, row in log_excess.iterrows():
        if float(row["target_position"]) < 1.0 - 1e-12:
            active_rows.append(index)
        elif active_rows:
            frame = log_excess.loc[active_rows]
            cycles.append(
                {
                    "start_date": str(frame["date"].iloc[0].date()),
                    "end_date": str(frame["date"].iloc[-1].date()),
                    "log_excess": float(frame["log_excess"].sum()),
                    "complete": True,
                }
            )
            active_rows = []
    if active_rows:
        frame = log_excess.loc[active_rows]
        cycles.append(
            {
                "start_date": str(frame["date"].iloc[0].date()),
                "end_date": str(frame["date"].iloc[-1].date()),
                "log_excess": float(frame["log_excess"].sum()),
                "complete": False,
            }
        )
    total = float(log_excess["log_excess"].sum())
    if total <= 0.0:
        return None, cycles
    maximum_positive = max(
        (float(cycle["log_excess"]) for cycle in cycles if cycle["log_excess"] > 0.0),
        default=0.0,
    )
    return float(maximum_positive / total), cycles


def evaluate_portfolios(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    event_report: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """事件门通过后仅运行冻结5Y、3Y与双倍成本路径。"""

    if not event_report["passed"]:
        raise PermissionError("事件门未通过，组合回测不允许运行")
    five = _evaluate_model(
        panel, market, dividends, benchmark, config, years=5, cost_scenario="base"
    )
    three = _evaluate_model(
        panel, market, dividends, benchmark, config, years=3, cost_scenario="base"
    )
    double = _evaluate_model(
        panel, market, dividends, benchmark, config, years=5, cost_scenario="double"
    )

    historical = config["historical_gates"]
    robustness = config["robustness_gates"]
    main_gates = {
        "five_year_net_sharpe_at_least_1_20": bool(
            five["summary"]["sharpe_zero_cash_rate"] is not None
            and five["summary"]["sharpe_zero_cash_rate"]
            >= float(historical["five_year_net_sharpe_minimum"])
        ),
        "five_year_excess_vs_h00300_positive": five["annualized_excess_vs_h00300"] > 0.0,
        "five_year_excess_vs_buy_hold_positive": five["annualized_excess_vs_buy_hold"] > 0.0,
        "max_drawdown_at_most_75_percent_of_buy_hold": bool(
            five["max_drawdown_ratio_vs_buy_hold"] is not None
            and five["max_drawdown_ratio_vs_buy_hold"]
            <= float(historical["max_drawdown_ratio_vs_buy_hold_maximum"])
        ),
        "complete_round_trips_per_year_at_most_10": (
            five["maximum_complete_round_trips_in_any_year"]
            <= int(historical["maximum_complete_round_trips_per_year"])
        ),
    }

    leave_one_year_out = _leave_one_year_out_sharpes(five["ledger"])
    log_excess = _aligned_log_excess(five["ledger"], five["buy_hold_ledger"])
    year_share, yearly_contributions = _single_year_excess_share(log_excess)
    cycle_share, cycle_contributions = _single_risk_cycle_excess_share(log_excess)
    robustness_gates = {
        "three_year_net_sharpe_at_least_0_90": bool(
            three["summary"]["sharpe_zero_cash_rate"] is not None
            and three["summary"]["sharpe_zero_cash_rate"]
            >= float(robustness["three_year_net_sharpe_minimum"])
        ),
        "three_year_excess_vs_both_benchmarks_positive": bool(
            three["annualized_excess_vs_h00300"] > 0.0
            and three["annualized_excess_vs_buy_hold"] > 0.0
        ),
        "double_cost_five_year_sharpe_at_least_0_90": bool(
            double["summary"]["sharpe_zero_cash_rate"] is not None
            and double["summary"]["sharpe_zero_cash_rate"]
            >= float(robustness["double_cost_five_year_sharpe_minimum"])
        ),
        "leave_one_calendar_year_out_sharpe_positive": bool(
            leave_one_year_out
            and all(value is not None and value > 0.0 for value in leave_one_year_out.values())
        ),
        "single_year_log_excess_share_at_most_50_percent": bool(
            year_share is not None
            and year_share
            <= float(robustness["maximum_single_year_share_of_total_log_excess"])
        ),
        "single_risk_cycle_log_excess_share_at_most_50_percent": bool(
            cycle_share is not None
            and cycle_share
            <= float(robustness["maximum_single_risk_cycle_share_of_total_log_excess"])
        ),
        "minimum_eight_independent_events": bool(
            event_report["independent_complete_primary_events"]
            >= int(robustness["minimum_independent_events"])
        ),
    }
    passed = all(main_gates.values()) and all(robustness_gates.values())
    decision = (
        "HISTORICAL_CANDIDATE_PASS"
        if passed
        else "REJECTED_FROZEN_HISTORICAL_OR_ROBUSTNESS_GATE_FAILED_NO_RESCUE"
    )

    def compact(model: dict[str, Any]) -> dict[str, Any]:
        return {
            "years": model["years"],
            "cost_scenario": model["cost_scenario"],
            "strategy": model["summary"],
            "buy_hold": model["buy_hold_summary"],
            "h00300": model["h00300_summary"],
            "annualized_excess_vs_buy_hold": model["annualized_excess_vs_buy_hold"],
            "annualized_excess_vs_h00300": model["annualized_excess_vs_h00300"],
            "max_drawdown_ratio_vs_buy_hold": model["max_drawdown_ratio_vs_buy_hold"],
            "complete_round_trips_by_year": model["complete_round_trips_by_year"],
            "maximum_complete_round_trips_in_any_year": model[
                "maximum_complete_round_trips_in_any_year"
            ],
        }

    report = {
        "decision": decision,
        "passed": passed,
        "five_year": compact(five),
        "three_year": compact(three),
        "five_year_double_cost": compact(double),
        "leave_one_year_out_sharpes": leave_one_year_out,
        "yearly_log_excess_contributions_vs_buy_hold": yearly_contributions,
        "maximum_single_year_share_of_total_log_excess": year_share,
        "risk_cycle_log_excess_contributions_vs_buy_hold": cycle_contributions,
        "maximum_single_risk_cycle_share_of_total_log_excess": cycle_share,
        "main_gates": main_gates,
        "robustness_gates": robustness_gates,
        "failed_gates": [
            name
            for name, passed_gate in {**main_gates, **robustness_gates}.items()
            if not passed_gate
        ],
    }
    artifacts = {
        "five_year_ledger": five["ledger"],
        "five_year_trades": five["trades"],
        "three_year_ledger": three["ledger"],
        "three_year_trades": three["trades"],
        "double_cost_ledger": double["ledger"],
        "double_cost_trades": double["trades"],
        "buy_hold_ledger": five["buy_hold_ledger"],
        "buy_hold_trades": five["buy_hold_trades"],
    }
    return report, artifacts


def _read_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    contracts = config["data_contracts"]
    return {
        "fdr": pd.read_parquet(ROOT / contracts["fdr007"]["file"]),
        "fx": pd.read_parquet(ROOT / contracts["usdcny_midpoint"]["file"]),
        "pmi": pd.read_parquet(ROOT / contracts["pmi_new_orders"]["file"]),
        "tsf": pd.read_parquet(ROOT / contracts["tsf_stock_yoy"]["file"]),
        "market": pd.read_parquet(ROOT / contracts["etf_market"]["file"]),
        "benchmark": pd.read_parquet(ROOT / contracts["benchmark_total_return"]["file"]),
        "dividends": pd.read_csv(ROOT / contracts["etf_dividends"]["file"]),
    }


def evaluate_protocol(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """执行冻结研究；事件门失败时保持组合回测为SKIPPED。"""

    inputs = _read_inputs(config)
    panel = build_point_in_time_panel(
        inputs["market"],
        inputs["fdr"],
        inputs["fx"],
        inputs["pmi"],
        inputs["tsf"],
        config,
    )
    event_report, events, controls = evaluate_event_gate(
        panel,
        inputs["dividends"],
        inputs["benchmark"],
        config,
    )
    artifacts: dict[str, pd.DataFrame] = {
        "point_in_time_panel": panel,
        "event_table": events,
        "control_table": controls,
    }
    if event_report["passed"]:
        portfolio_report, portfolio_artifacts = evaluate_portfolios(
            panel,
            inputs["market"],
            inputs["dividends"],
            inputs["benchmark"],
            event_report,
            config,
        )
        artifacts.update(portfolio_artifacts)
        decision = portfolio_report["decision"]
        return_evaluation = "COMPLETED_ONCE"
        portfolio_status = "COMPLETED_ONCE_AFTER_EVENT_GATE_PASS"
    else:
        portfolio_report = {
            "decision": "SKIPPED_EVENT_GATE_FAILED",
            "passed": False,
            "metrics": None,
            "failed_gates": event_report["failed_gates"],
        }
        decision = event_report["decision"]
        return_evaluation = "NOT_ALLOWED"
        portfolio_status = "SKIPPED_EVENT_GATE_FAILED"

    first_dates = {
        f"{years}y": (
            str(panel.loc[panel[f"model_eligible_{years}y"], "date"].iloc[0].date())
            if panel[f"model_eligible_{years}y"].any()
            else None
        )
        for years in (3, 5)
    }
    report = {
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "decision": decision,
        "data_scope": {
            "start": config["protocol"]["data_start"],
            "end": config["protocol"]["data_end"],
            "pre_2021_data_used": False,
            "first_model_eligible_dates": first_dates,
            "strict_full_calendar_windows": True,
            "expanding_or_shortened_windows_used": False,
        },
        "event_study": event_report,
        "portfolio_backtest": portfolio_report,
        "return_evaluation": return_evaluation,
        "portfolio_backtest_status": portfolio_status,
        "no_parameter_rescue": True,
        "forward_shadow_required": bool(config["governance"]["forward_shadow_required"]),
        "minimum_new_shadow_trading_days": int(
            config["governance"]["minimum_new_shadow_trading_days"]
        ),
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
    }
    return report, artifacts


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"无法JSON序列化：{type(value).__name__}")


def markdown_report(report: dict[str, Any]) -> str:
    """生成保持状态语义的中文研究报告。"""

    event = report["event_study"]
    lines = [
        "# 510300宏观压力规避V1冻结研究结果",
        "",
        f"- 最终状态：`{report['decision']}`",
        f"- 收益评估：`{report['return_evaluation']}`",
        f"- 组合回测：`{report['portfolio_backtest_status']}`",
        "- 数据范围：2021-01-01至2026-08-25，未使用2016—2020数据",
        "- 资产边界：仅510300.SH与人民币现金",
        "- 实盘授权：`false`",
        "",
        "## 严格滚动窗口可用性",
        "",
        f"- 3年模型首个有效信号日：{report['data_scope']['first_model_eligible_dates']['3y']}",
        f"- 5年模型首个有效信号日：{report['data_scope']['first_model_eligible_dates']['5y']}",
        "- 当期观察不进入90%分位；未使用扩展窗口或缩短窗口",
        "",
        "## 5年主模型事件门",
        "",
        f"- 有效交易日：{event['eligible_trading_days']}",
        f"- 原始事件起点：{event['raw_onsets']}",
        f"- 间隔20交易日后的独立事件：{event['independent_onsets']}",
        f"- 具有完整20日结果的独立事件：{event['independent_complete_primary_events']}",
        f"- 事件门状态：`{event['decision']}`",
        "",
        "| 事件硬门 | 通过 |",
        "|---|---:|",
    ]
    for name, passed in event["gates"].items():
        lines.append(f"| {name} | {'是' if passed else '否'} |")
    lines.extend(["", "## 组合层", ""])
    if report["return_evaluation"] == "NOT_ALLOWED":
        lines.extend(
            [
                "事件门未通过，因此未运行5年、3年或双倍成本组合回测；",
                "没有可报告的净夏普率、净超额收益或最大回撤。该状态不能解释为零收益或亏损。",
            ]
        )
    else:
        portfolio = report["portfolio_backtest"]
        five = portfolio["five_year"]
        three = portfolio["three_year"]
        double = portfolio["five_year_double_cost"]
        lines.extend(
            [
                f"- 5年净夏普率：{five['strategy']['sharpe_zero_cash_rate']:.6f}",
                f"- 5年相对H00300年化超额：{five['annualized_excess_vs_h00300']:.6%}",
                f"- 5年相对510300含分红买入持有年化超额：{five['annualized_excess_vs_buy_hold']:.6%}",
                f"- 3年净夏普率：{three['strategy']['sharpe_zero_cash_rate']:.6f}",
                f"- 5年双倍成本净夏普率：{double['strategy']['sharpe_zero_cash_rate']:.6f}",
                "",
                f"组合判定：`{portfolio['decision']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 治理边界",
            "",
            "- 参数营救：禁止",
            "- 仓位映射到真实账户：禁用",
            "- 订单生成：禁用",
            "- 券商连接：禁用",
            "- 即使历史通过，也必须新增252个交易日影子前向后再评估",
            "",
        ]
    )
    return "\n".join(lines)
