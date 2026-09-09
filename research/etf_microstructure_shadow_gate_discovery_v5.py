"""510300二元仓位的ETF微观结构共识与242日影子失效门开发研究。

本脚本只生成研究证据，不生成订单，也不连接券商。唯一执行资产是510300，
目标状态严格为满仓或空仓。所有特征在T日收盘后形成，仓位在T+1开盘执行。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import global_liquidity_regime_5d_discovery_v0 as market_loader
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_SHADOW_GATE_DISCOVERY_V5"
CANDIDATE_POLICY = "MICRO_CONSENSUS3_OR_SHORT_EXTREMES_DUAL_SHADOW"
DATA_CUTOFF = pd.Timestamp("2026-08-14")
SIGNAL_DATA_CUTOFF = pd.Timestamp("2026-08-12")

NAV_FILE = PROJECT_ROOT / "data" / "raw" / "fund" / "510300_nav_daily_raw.parquet"
FUND_SHARE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "flow" / "510300_fund_share_daily_tushare.parquet"
)
MARGIN_FILE = (
    PROJECT_ROOT / "data" / "raw" / "flow" / "510300_margin_detail_daily_tushare.parquet"
)
ETF_FILE = PROJECT_ROOT / "data" / "raw" / "r6" / "510300_daily.parquet"
BENCHMARK_FILE = PROJECT_ROOT / "data" / "raw" / "r6" / "H00300_total_return_daily.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"

OUTPUT_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "510300_etf_microstructure_shadow_gate_discovery_v5.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_etf_microstructure_shadow_gate_discovery_v5"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "discovery"
    / "510300_etf_microstructure_shadow_gate_discovery_v5.json"
)

RANK_WINDOW = 252
RANK_MINIMUM_ROWS = 126
SHADOW_WINDOW = 242
SHADOW_MINIMUM_MATURED_ROWS = 20
SHORT_SHADOW_WINDOW = 60
SHORT_SHADOW_MINIMUM_MATURED_ROWS = 10
SHORT_SHADOW_FLOOR = -0.01
MATURITY_LAG_SIGNAL_ROWS = 2
CASH_ANNUAL_RATE = 0.015
TRADING_DAYS_PER_YEAR = 242
SHADOW_STRESS_COST_PER_TRANSITION = 0.0016
START_PERTURBATIONS = [0, 1, 2, 3, 4]

PERIODS = {
    "DEVELOPMENT_2022_2025": (pd.Timestamp("2022-03-09"), pd.Timestamp("2025-12-31")),
    "TEMPORAL_REPLICATION_2026_YTD": (
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-08-14"),
    ),
    "COMBINED_HISTORY_TO_2026_08_14": (
        pd.Timestamp("2022-03-09"),
        pd.Timestamp("2026-08-14"),
    ),
}

POLICIES = [
    "MICRO_CONSENSUS3_ONLY",
    "MICRO_CONSENSUS3_OR_SHORT_EXTREMES_RAW",
    CANDIDATE_POLICY,
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        raise ValueError(f"{column}存在重复日期")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少字段：{missing}")


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_paths = [
        NAV_FILE,
        FUND_SHARE_FILE,
        MARGIN_FILE,
        ETF_FILE,
        BENCHMARK_FILE,
        DIVIDEND_FILE,
    ]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少研究输入：{missing}")

    nav = _normalize_date(pd.read_parquet(NAV_FILE))
    shares = _normalize_date(pd.read_parquet(FUND_SHARE_FILE))
    margin = _normalize_date(pd.read_parquet(MARGIN_FILE))
    etf = _normalize_date(pd.read_parquet(ETF_FILE))
    benchmark = _normalize_date(pd.read_parquet(BENCHMARK_FILE))
    dividends = pd.read_csv(DIVIDEND_FILE)
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)

    _require_columns(nav, {"date", "close_premium_to_nav"}, "净值数据")
    _require_columns(shares, {"date", "fund_shares"}, "基金份额数据")
    _require_columns(
        margin,
        {"date", "rzye", "rqye", "rqyl", "rzrqye"},
        "融资融券数据",
    )
    _require_columns(etf, {"date", "open", "close"}, "510300行情")
    _require_columns(benchmark, {"date", "close"}, "H00300全收益基准")

    signal_start = max(nav["date"].min(), shares["date"].min(), margin["date"].min())
    signal = etf.loc[
        etf["date"].between(signal_start, SIGNAL_DATA_CUTOFF), ["date"]
    ].copy()
    signal = (
        signal.merge(
            nav[["date", "close_premium_to_nav"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            shares[["date", "fund_shares"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            margin[["date", "rzye", "rqye", "rqyl", "rzrqye"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    signal_value_columns = [
        "close_premium_to_nav",
        "fund_shares",
        "rzye",
        "rqye",
        "rqyl",
        "rzrqye",
    ]
    missing_signal_values = signal[signal_value_columns].isna().sum()
    if int(missing_signal_values.sum()) != 0:
        raise ValueError(f"交易日信号输入存在缺失：{missing_signal_values.to_dict()}")
    if signal.empty or signal["date"].max() != SIGNAL_DATA_CUTOFF:
        raise ValueError(
            f"信号数据截止日异常：实际{signal['date'].max()}，预期{SIGNAL_DATA_CUTOFF}"
        )
    if etf["date"].max() < DATA_CUTOFF or benchmark["date"].max() < DATA_CUTOFF:
        raise ValueError("执行行情或全收益基准未覆盖冻结数据截止日")

    market = etf.loc[etf["date"].le(DATA_CUTOFF), ["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    market = market.merge(
        benchmark.loc[benchmark["date"].le(DATA_CUTOFF), ["date", "close"]].rename(
            columns={"close": "benchmark_close"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("执行行情与H00300全收益基准对齐失败")

    audit = {
        "signal_rows": int(len(signal)),
        "signal_first_date": signal["date"].min().date().isoformat(),
        "signal_last_date": signal["date"].max().date().isoformat(),
        "market_rows_to_cutoff": int(len(market)),
        "market_first_date": market["date"].min().date().isoformat(),
        "market_last_date": market["date"].max().date().isoformat(),
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in input_paths
        },
    }
    return signal, dividends, {"market": market, "audit": audit}


def _rolling_last_percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def rank_last(values: np.ndarray) -> float:
        last = values[-1]
        if not np.isfinite(last):
            return np.nan
        finite = values[np.isfinite(values)]
        if len(finite) < RANK_MINIMUM_ROWS:
            return np.nan
        return float(np.mean(finite <= last))

    return numeric.rolling(RANK_WINDOW, min_periods=RANK_MINIMUM_ROWS).apply(
        rank_last, raw=True
    )


def _hysteresis(
    rank: pd.Series,
    *,
    side: str,
    entry: float,
    recovery: float,
) -> pd.Series:
    if side not in {"LOW", "HIGH"}:
        raise ValueError("滞回方向只能是LOW或HIGH")
    state = False
    output: list[bool] = []
    for value in pd.to_numeric(rank, errors="coerce"):
        if pd.notna(value):
            number = float(value)
            if not state:
                state = number <= entry if side == "LOW" else number >= entry
            else:
                recovered = number >= recovery if side == "LOW" else number <= recovery
                if recovered:
                    state = False
        output.append(state)
    return pd.Series(output, index=rank.index, dtype=bool)


def _build_open_to_open_target(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.Series:
    execution = market[["date", "etf_open"]].copy()
    dividend_by_ex_date = (
        dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    )
    next_open = execution["etf_open"].shift(-1)
    following_open = execution["etf_open"].shift(-2)
    following_date = execution["date"].shift(-2)
    following_dividend = following_date.map(
        lambda value: float(dividend_by_ex_date.get(value, 0.0))
        if pd.notna(value)
        else 0.0
    )
    execution["one_day_open_total_return"] = (
        following_open + following_dividend
    ) / next_open - 1.0
    mapped = signal[["date"]].merge(
        execution[["date", "one_day_open_total_return"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    return mapped["one_day_open_total_return"]


def _add_rule_features(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    frame = signal.copy()
    premium = pd.to_numeric(frame["close_premium_to_nav"], errors="coerce")
    frame["premium_std20"] = premium.rolling(20, min_periods=20).std(ddof=0)
    frame["premium_range20"] = (
        premium.rolling(20, min_periods=20).max()
        - premium.rolling(20, min_periods=20).min()
    )
    frame["premium_std10"] = premium.rolling(10, min_periods=10).std(ddof=0)

    for source, horizon in [
        ("fund_shares", 20),
        ("rzye", 40),
        ("rzrqye", 3),
        ("rqye", 20),
    ]:
        values = pd.to_numeric(frame[source], errors="coerce")
        logged = np.log(values.where(values.gt(0.0)))
        frame[f"{source}_log_change{horizon}"] = logged - logged.shift(horizon)

    frame["short_inventory_scale20"] = (
        pd.to_numeric(frame["rqyl"], errors="coerce").rolling(20, min_periods=20).sum()
        / pd.to_numeric(frame["rzrqye"], errors="coerce").replace(0.0, np.nan)
    )

    rank_sources = [
        "premium_std20",
        "premium_range20",
        "premium_std10",
        "fund_shares_log_change20",
        "rzye_log_change40",
        "rzrqye_log_change3",
        "rqye_log_change20",
        "short_inventory_scale20",
    ]
    for source in rank_sources:
        frame[f"rank252_{source}"] = _rolling_last_percentile(frame[source])

    frame["risk_premium_std20"] = _hysteresis(
        frame["rank252_premium_std20"], side="LOW", entry=0.10, recovery=0.30
    )
    frame["risk_premium_range20"] = _hysteresis(
        frame["rank252_premium_range20"], side="LOW", entry=0.15, recovery=0.30
    )
    frame["risk_premium_std10"] = _hysteresis(
        frame["rank252_premium_std10"], side="LOW", entry=0.20, recovery=0.60
    )
    frame["risk_fund_share_contraction"] = _hysteresis(
        frame["rank252_fund_shares_log_change20"],
        side="LOW",
        entry=0.15,
        recovery=0.70,
    )
    frame["risk_financing_expansion40"] = _hysteresis(
        frame["rank252_rzye_log_change40"], side="HIGH", entry=0.85, recovery=0.50
    )
    frame["risk_total_margin_shock3"] = _hysteresis(
        frame["rank252_rzrqye_log_change3"], side="LOW", entry=0.02, recovery=0.95
    )
    frame["risk_short_balance_contraction20"] = frame[
        "rank252_rqye_log_change20"
    ].le(0.08)
    frame["risk_short_inventory_drought"] = frame[
        "rank252_short_inventory_scale20"
    ].le(0.01)
    frame["risk_short_inventory_crowding"] = _hysteresis(
        frame["rank252_short_inventory_scale20"],
        side="HIGH",
        entry=0.95,
        recovery=0.70,
    )

    vote_columns = [
        "risk_premium_std20",
        "risk_premium_range20",
        "risk_premium_std10",
        "risk_fund_share_contraction",
        "risk_financing_expansion40",
        "risk_total_margin_shock3",
        "risk_short_balance_contraction20",
    ]
    frame["micro_risk_vote_count"] = frame[vote_columns].sum(axis=1).astype(np.int8)
    frame["cash_consensus3"] = frame["micro_risk_vote_count"].ge(3)
    frame["cash_raw"] = (
        frame["cash_consensus3"]
        | frame["risk_short_inventory_drought"]
        | frame["risk_short_inventory_crowding"]
    )

    frame["one_day_open_total_return"] = _build_open_to_open_target(
        frame, market, dividends
    )
    raw_transition = frame["cash_raw"].ne(frame["cash_raw"].shift(1).fillna(False))
    frame["raw_transition"] = raw_transition.astype(bool)
    cash_daily = CASH_ANNUAL_RATE / TRADING_DAYS_PER_YEAR
    frame["raw_shadow_net_contribution"] = np.where(
        frame["cash_raw"] & frame["one_day_open_total_return"].notna(),
        cash_daily - frame["one_day_open_total_return"],
        0.0,
    ) - frame["raw_transition"].astype(float) * SHADOW_STRESS_COST_PER_TRANSITION

    matured_shadow = frame["raw_shadow_net_contribution"].shift(MATURITY_LAG_SIGNAL_ROWS)
    frame["shadow242_net_sum"] = matured_shadow.rolling(
        SHADOW_WINDOW,
        min_periods=SHADOW_MINIMUM_MATURED_ROWS,
    ).sum()
    frame["shadow60_net_sum"] = matured_shadow.rolling(
        SHORT_SHADOW_WINDOW,
        min_periods=SHORT_SHADOW_MINIMUM_MATURED_ROWS,
    ).sum()
    annual_gate = frame["shadow242_net_sum"].gt(0.0) | frame["shadow242_net_sum"].isna()
    short_gate = frame["shadow60_net_sum"].gt(SHORT_SHADOW_FLOOR) | frame[
        "shadow60_net_sum"
    ].isna()
    frame["shadow_gate_active"] = annual_gate & short_gate
    frame["cash_final"] = frame["cash_raw"] & frame["shadow_gate_active"]
    frame["state_final"] = np.where(frame["cash_final"], 0, 1).astype(np.int8)
    return frame


def _execution_frame(market: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "cash_consensus3", "cash_raw", "cash_final"]
    frame = market.merge(features[columns], on="date", how="left", validate="one_to_one")
    for policy, source in [
        ("MICRO_CONSENSUS3_ONLY", "cash_consensus3"),
        ("MICRO_CONSENSUS3_OR_SHORT_EXTREMES_RAW", "cash_raw"),
        (CANDIDATE_POLICY, "cash_final"),
    ]:
        frame[f"{policy}_cash_execution"] = frame[source].shift(1).fillna(False).astype(bool)
    return frame


def _evaluate_policy_period(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy: str,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    selection = np.flatnonzero(frame["date"].between(start, end).to_numpy())
    if len(selection) <= max(START_PERTURBATIONS):
        raise ValueError(f"{period_name}样本不足以执行起点扰动")
    last_execution_index = int(selection[-1])
    cash_column = f"{policy}_cash_execution"
    rows: list[dict[str, Any]] = []
    for perturbation in START_PERTURBATIONS:
        first_execution_index = int(selection[perturbation])
        anchor_index = first_execution_index - 1
        sample = frame.iloc[anchor_index : last_execution_index + 1][
            ["date", "etf_open", "etf_close", "benchmark_close"]
        ].reset_index(drop=True)
        desired_states = np.ones(len(sample), dtype=np.int8)
        desired_states[0] = 0
        desired_states[1:] = np.where(
            frame.loc[first_execution_index:last_execution_index, cash_column].to_numpy(bool),
            0,
            1,
        )
        benchmark = build_benchmark_ledger(sample, market_loader.INITIAL_CAPITAL_CNY)
        base_ledger, base_trades = simulate_binary_path(
            sample,
            dividends,
            desired_states,
            costs=market_loader.BASE_COSTS,
            initial_capital=market_loader.INITIAL_CAPITAL_CNY,
            reinvest_paid_dividends=True,
        )
        stress_ledger, stress_trades = simulate_binary_path(
            sample,
            dividends,
            desired_states,
            costs=market_loader.STRESS_COSTS,
            initial_capital=market_loader.INITIAL_CAPITAL_CNY,
            reinvest_paid_dividends=True,
        )
        base = summarize_path(
            base_ledger,
            base_trades,
            benchmark,
            objective=market_loader.OBJECTIVE,
        )
        stress = summarize_path(
            stress_ledger,
            stress_trades,
            benchmark,
            objective=market_loader.OBJECTIVE,
        )
        row: dict[str, Any] = {
            "policy": policy,
            "period": period_name,
            "start_perturbation": perturbation,
            "first_execution_date": frame.loc[first_execution_index, "date"],
            "last_execution_date": frame.loc[last_execution_index, "date"],
            "execution_day_count": int(last_execution_index - first_execution_index + 1),
            "cash_day_count": int((desired_states[1:] == 0).sum()),
            "cash_day_share": float((desired_states[1:] == 0).mean()),
        }
        for prefix, summary in [("base", base), ("stress", stress)]:
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for (policy, period), group in metrics.groupby(["policy", "period"], sort=True):
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        base_annual = pd.to_numeric(group["base_annualized_excess"], errors="coerce")
        valid_rolling = rolling.dropna()
        every_pass = bool(group["stress_both_20pct_gates"].fillna(False).astype(bool).all())
        aggregates.append(
            {
                "policy": policy,
                "period": period,
                "base_annualized_excess_minimum": _safe_float(base_annual.min()),
                "base_annualized_excess_median": _safe_float(base_annual.median()),
                "stress_annualized_excess_minimum": _safe_float(annual.min()),
                "stress_annualized_excess_median": _safe_float(annual.median()),
                "stress_annualized_excess_maximum": _safe_float(annual.max()),
                "stress_rolling_242d_excess_minimum": _safe_float(valid_rolling.min()),
                "stress_rolling_242d_excess_median": _safe_float(valid_rolling.median()),
                "cash_day_share_median": _safe_float(group["cash_day_share"].median()),
                "stress_trade_leg_count_median": _safe_float(
                    group["stress_trade_leg_count"].median()
                ),
                "stress_start_pass_ratio": float(
                    group["stress_both_20pct_gates"].fillna(False).astype(bool).mean()
                ),
                "every_start_both_20pct_gates": every_pass,
            }
        )
    return aggregates


def _calendar_shadow_contribution(features: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = features.loc[
        features["date"].between(PERIODS["COMBINED_HISTORY_TO_2026_08_14"][0], SIGNAL_DATA_CUTOFF)
    ].copy()
    eligible["year"] = eligible["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in eligible.groupby("year", sort=True):
        final_transition = group["cash_final"].ne(group["cash_final"].shift(1).fillna(False))
        contribution = np.where(
            group["cash_final"] & group["one_day_open_total_return"].notna(),
            CASH_ANNUAL_RATE / TRADING_DAYS_PER_YEAR - group["one_day_open_total_return"],
            0.0,
        ) - final_transition.astype(float) * SHADOW_STRESS_COST_PER_TRANSITION
        rows.append(
            {
                "year": int(year),
                "signal_rows": int(len(group)),
                "cash_signal_days": int(group["cash_final"].sum()),
                "approximate_stress_net_cash_advantage_sum": float(np.sum(contribution)),
                "last_shadow242_net_sum": _safe_float(group["shadow242_net_sum"].iloc[-1]),
                "last_shadow60_net_sum": _safe_float(group["shadow60_net_sum"].iloc[-1]),
                "last_shadow_gate_active": bool(group["shadow_gate_active"].iloc[-1]),
            }
        )
    return rows


def main() -> int:
    signal, dividends, loaded = _read_inputs()
    market = loaded["market"]
    features = _add_rule_features(signal, market, dividends)
    execution = _execution_frame(market, features)

    metric_frames: list[pd.DataFrame] = []
    for policy in POLICIES:
        for period_name, (start, end) in PERIODS.items():
            metric_frames.append(
                _evaluate_policy_period(
                    execution,
                    dividends,
                    policy,
                    period_name,
                    start,
                    end,
                )
            )
    metrics = pd.concat(metric_frames, ignore_index=True)
    aggregates = _aggregate(metrics)

    candidate_combined = next(
        row
        for row in aggregates
        if row["policy"] == CANDIDATE_POLICY
        and row["period"] == "COMBINED_HISTORY_TO_2026_08_14"
    )
    candidate_development = next(
        row
        for row in aggregates
        if row["policy"] == CANDIDATE_POLICY
        and row["period"] == "DEVELOPMENT_2022_2025"
    )
    candidate_temporal = next(
        row
        for row in aggregates
        if row["policy"] == CANDIDATE_POLICY
        and row["period"] == "TEMPORAL_REPLICATION_2026_YTD"
    )
    historical_gate = bool(
        candidate_combined["every_start_both_20pct_gates"]
        and candidate_development["every_start_both_20pct_gates"]
    )
    status = (
        "HISTORICAL_CANDIDATE_FOUND_FORWARD_SHADOW_REQUIRED_NOT_TRADING_AUTHORIZED"
        if historical_gate
        else "HISTORICAL_CANDIDATE_REJECTED_CONTINUE_SEARCH"
    )

    output_columns = [
        "date",
        "close_premium_to_nav",
        "premium_std20",
        "premium_range20",
        "premium_std10",
        "fund_shares_log_change20",
        "rzye_log_change40",
        "rzrqye_log_change3",
        "rqye_log_change20",
        "short_inventory_scale20",
        "rank252_premium_std20",
        "rank252_premium_range20",
        "rank252_premium_std10",
        "rank252_fund_shares_log_change20",
        "rank252_rzye_log_change40",
        "rank252_rzrqye_log_change3",
        "rank252_rqye_log_change20",
        "rank252_short_inventory_scale20",
        "risk_premium_std20",
        "risk_premium_range20",
        "risk_premium_std10",
        "risk_fund_share_contraction",
        "risk_financing_expansion40",
        "risk_total_margin_shock3",
        "risk_short_balance_contraction20",
        "risk_short_inventory_drought",
        "risk_short_inventory_crowding",
        "micro_risk_vote_count",
        "cash_consensus3",
        "cash_raw",
        "one_day_open_total_return",
        "raw_transition",
        "raw_shadow_net_contribution",
        "shadow242_net_sum",
        "shadow60_net_sum",
        "shadow_gate_active",
        "cash_final",
        "state_final",
    ]
    _atomic_parquet(features[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(metrics, OUTPUT_METRICS)

    payload = {
        "project_id": PROJECT_ID,
        "status": status,
        "candidate_policy": CANDIDATE_POLICY,
        "historical_gate": historical_gate,
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "allowed_states": [0, 1],
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "benchmark": "H00300_TOTAL_RETURN",
            "derivatives_execution": False,
        },
        "objective": {
            "minimum_annualized_net_excess": 0.20,
            "minimum_rolling_242d_excess_median": 0.20,
            "every_start_perturbation_must_pass": True,
            "start_perturbations": START_PERTURBATIONS,
            "initial_capital_cny": market_loader.INITIAL_CAPITAL_CNY,
            "cash_annual_rate": CASH_ANNUAL_RATE,
            "base_slippage_bps_per_leg": 5.0,
            "stress_slippage_bps_per_leg": 15.0,
            "commission_bps_per_leg": 1.0,
            "lot_size_shares": 100,
        },
        "rule": {
            "rank_window_rows": RANK_WINDOW,
            "rank_minimum_rows": RANK_MINIMUM_ROWS,
            "micro_consensus_required_votes": 3,
            "micro_domains": {
                "premium_std20": "低10%进入、回升至30%退出",
                "premium_range20": "低15%进入、回升至30%退出",
                "premium_std10": "低20%进入、回升至60%退出",
                "fund_shares_log_change20": "低15%进入、回升至70%退出",
                "financing_balance_log_change40": "高85%进入、回落至50%退出",
                "total_margin_balance_log_change3": "低2%进入、回升至95%退出",
                "short_balance_log_change20": "低8%当日触发",
            },
            "short_inventory_extremes": {
                "scale": "20日融券余量合计/当日融资融券余额",
                "drought": "252日滚动分位不高于1%时当日触发",
                "crowding": "252日滚动分位达到95%进入，回落至70%退出",
            },
            "raw_cash": "微观风险票数至少3票，或融券库存枯竭/拥挤任一触发",
            "shadow_gate": {
                "window_matured_signal_rows": SHADOW_WINDOW,
                "minimum_matured_rows": SHADOW_MINIMUM_MATURED_ROWS,
                "short_window_matured_signal_rows": SHORT_SHADOW_WINDOW,
                "short_window_minimum_matured_rows": SHORT_SHADOW_MINIMUM_MATURED_ROWS,
                "short_window_net_floor": SHORT_SHADOW_FLOOR,
                "maturity_lag_signal_rows": MATURITY_LAG_SIGNAL_ROWS,
                "stress_cost_per_raw_transition": SHADOW_STRESS_COST_PER_TRANSITION,
                "active": (
                    "242日已成熟原始规则影子净贡献之和大于0，且60日净贡献大于-1%；"
                    "相应窗口观测不足时该门保持开启"
                ),
            },
            "final_cash": "原始空仓条件成立且242日影子失效门开启",
        },
        "periods": {
            name: {"start": start.date().isoformat(), "end": end.date().isoformat()}
            for name, (start, end) in PERIODS.items()
        },
        "aggregates": aggregates,
        "candidate_key_results": {
            "development_2022_2025": candidate_development,
            "temporal_replication_2026_ytd": candidate_temporal,
            "combined_history_to_2026_08_14": candidate_combined,
        },
        "calendar_shadow_contribution": _calendar_shadow_contribution(features),
        "input_audit": loaded["audit"],
        "evidence": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)),
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
            "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        },
        "boundaries": {
            "all_rows_through_2026_08_14_were_read_during_discovery": True,
            "2026_ytd_is_temporal_replication_not_unseen_validation": True,
            "post_freeze_unseen_forward_days": 0,
            "deployable_strategy": False,
            "paper_or_shadow_signal_authorized": False,
            "broker_connection": "DISABLED",
            "order_generation": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if historical_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
