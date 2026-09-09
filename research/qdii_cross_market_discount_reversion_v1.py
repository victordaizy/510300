"""QDII跨市场合成折价回归V1的冻结研究实现。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import (
    DataContractError,
    evaluate_historical_returns,
    load_benchmark_series,
    load_etf_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "qdii_cross_market_discount_reversion_v1.yaml"


@dataclass
class QdiiPosition:
    """基础和压力现金账本共享的QDII ETF实物持仓。"""

    con_code: str
    group_id: str
    shares: int
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    entry_total_return_open: float
    entry_gross_notional: float
    prior_median_amount_20: float
    signal_date: pd.Timestamp
    discount_zscore: float
    relative_log_deviation: float
    last_total_return_close: float
    delayed_exit_days: int = 0


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取并校验候选合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("QDII折价回归合同必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    """阻止收益目标、夏普、成本和时点规则被静默弱化。"""

    failures: list[str] = []
    protocol = contract.get("protocol", {})
    account = contract.get("account", {})
    timing = contract.get("information_timing", {})
    signal = contract.get("signal", {})
    portfolio = contract.get("portfolio", {})
    costs = contract.get("costs", {})
    risk = contract.get("risk", {})
    gates = contract.get("visible_gates", {})
    safety = contract.get("safety", {})
    if protocol.get("candidate_id") != "QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V2":
        failures.append("parent_protocol_id")
    if protocol.get("lane") != "CROSS_BORDER_QDII_RELATIVE_VALUE":
        failures.append("lane")
    expected_numbers = {
        "initial_capital_cny": (account, 500000.0),
        "user_transaction_fee_rate_per_leg": (account, 0.0001),
        "minimum_annualized_net_excess": (gates, 0.40),
        "minimum_strategy_net_sharpe": (gates, 1.50),
        "rolling_equilibrium_lookback_trading_days": (signal, 120.0),
        "maximum_entry_zscore": (signal, -2.0),
        "maximum_entry_log_deviation": (signal, -0.01),
        "fixed_holding_period_trading_days_open_to_open": (signal, 5.0),
        "etf_user_fee_rate_per_leg": (costs, 0.0001),
    }
    for name, (section, expected) in expected_numbers.items():
        if float(section.get(name, np.nan)) != expected:
            failures.append(name)
    if gates.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    if timing.get("signal_information_time") != "CHINA_SIGNAL_DAY_CLOSE":
        failures.append("signal_information_time")
    if timing.get("first_executable_time") != "NEXT_CHINA_BENCHMARK_TRADING_DAY_OPEN":
        failures.append("first_executable_time")
    if timing.get("foreign_index_asof_rule") != "FOREIGN_MARKET_DATE_STRICTLY_BEFORE_CHINA_SIGNAL_DATE":
        failures.append("foreign_index_asof_rule")
    if int(timing.get("maximum_foreign_age_calendar_days", 0)) != 7:
        failures.append("maximum_foreign_age_calendar_days")
    if int(timing.get("maximum_fx_age_calendar_days", 0)) != 7:
        failures.append("maximum_fx_age_calendar_days")
    if bool(timing.get("same_open_price_used_for_signal_and_execution", True)):
        failures.append("same_open_price_used_for_signal_and_execution")
    if not bool(portfolio.get("enter_only_when_flat")):
        failures.append("enter_only_when_flat")
    if int(portfolio.get("maximum_positions_per_cycle", 0)) != 2:
        failures.append("maximum_positions_per_cycle")
    coverage = contract.get("prefreeze_coverage_gate", {})
    if int(coverage.get("minimum_executable_signal_rows", 0)) != 30:
        failures.append("minimum_executable_signal_rows")
    if int(coverage.get("minimum_signal_years", 0)) != 3:
        failures.append("minimum_signal_years")
    if int(coverage.get("minimum_tracking_groups_with_signals", 0)) != 2:
        failures.append("minimum_tracking_groups_with_signals")
    if not bool(costs.get("omitted_cost_is_failure")):
        failures.append("omitted_cost_is_failure")
    if any(
        bool(risk.get(key, True))
        for key in ("short_sale_allowed", "borrowing_allowed", "leverage_allowed")
    ):
        failures.append("risk_short_or_leverage")
    if any(
        bool(safety.get(key, True))
        for key in (
            "paper_position_generation",
            "shadow_signal_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety")
    groups = contract.get("tracked_groups", [])
    expected_groups = {
        "NASDAQ_100": ("NDX_daily.parquet", "USD/CNY", {"513100.SH", "159941.SZ"}),
        "SP_500": ("GSPC_daily.parquet", "USD/CNY", {"513500.SH"}),
        "DAX": ("GDAXI_daily.parquet", "EUR/CNY", {"513030.SH"}),
        "CAC_40": ("FCHI_daily.parquet", "EUR/CNY", {"513080.SH"}),
    }
    actual_groups = {
        str(item.get("group_id")): (
            str(item.get("index_file")),
            str(item.get("fx_series")),
            set(str(code) for code in item.get("etf_codes", [])),
        )
        for item in groups
    }
    if actual_groups != expected_groups:
        failures.append("tracked_groups")
    if failures:
        raise ValueError(f"QDII折价回归合同被弱化或损坏：{sorted(set(failures))}")


def load_fx_series(directory: Path, pattern: str) -> pd.DataFrame:
    """解析中国外汇交易中心历史中间价JSON，只保留合同需要的币种。"""

    rows: list[dict[str, Any]] = []
    files = sorted(directory.glob(pattern))
    if not files:
        raise DataContractError("没有人民币中间价输入文件")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = payload.get("pages")
        if not isinstance(pages, list):
            raise DataContractError(f"人民币中间价JSON缺少pages：{path.name}")
        for page in pages:
            data = page.get("data", {})
            names = [str(value) for value in data.get("searchlist", [])]
            records = page.get("records", data.get("records", []))
            if "USD/CNY" not in names or "EUR/CNY" not in names:
                raise DataContractError(f"人民币中间价JSON缺少USD或EUR：{path.name}")
            for record in records:
                values = record.get("values", [])
                if len(values) != len(names):
                    raise DataContractError(f"人民币中间价字段数不一致：{path.name}")
                item = {name: value for name, value in zip(names, values, strict=True)}
                rows.append(
                    {
                        "date": record.get("date"),
                        "USD/CNY": item["USD/CNY"],
                        "EUR/CNY": item["EUR/CNY"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("USD/CNY", "EUR/CNY"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty or frame["date"].duplicated().any():
        raise DataContractError("人民币中间价为空或日期重复")
    if not np.isfinite(frame[["USD/CNY", "EUR/CNY"]].to_numpy(float)).all():
        raise DataContractError("人民币中间价含非有限值")
    if not frame[["USD/CNY", "EUR/CNY"]].gt(0.0).all().all():
        raise DataContractError("人民币中间价必须为正")
    return frame.reset_index(drop=True)


def load_global_index(path: Path, expected_symbol: str) -> pd.DataFrame:
    """读取单个境外指数并验证代码、日期和价格。"""

    frame = pd.read_parquet(path)
    required = {"date", "close", "symbol", "source"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"境外指数缺少字段：{path.name} {sorted(missing)}")
    data = frame[["date", "close", "symbol", "source"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    if data.empty or data["date"].duplicated().any():
        raise DataContractError(f"境外指数为空或日期重复：{path.name}")
    symbols = set(data["symbol"].astype(str).unique())
    if symbols != {expected_symbol}:
        raise DataContractError(f"境外指数代码不符：{path.name} {symbols}")
    if not np.isfinite(data["close"].to_numpy(float)).all() or not data["close"].gt(0.0).all():
        raise DataContractError(f"境外指数收盘无效：{path.name}")
    return data.reset_index(drop=True)


def load_research_inputs(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """读取ETF、基金主表、基准、汇率和境外指数。"""

    inputs = contract["inputs"]
    panel, master = load_etf_inputs(
        ROOT / inputs["etf_total_return_panel"],
        ROOT / inputs["fund_master"],
    )
    benchmark = load_benchmark_series(
        ROOT / inputs["benchmark_total_return"],
        ROOT / inputs["benchmark_price_ohlc"],
    )
    fx = load_fx_series(
        ROOT / inputs["fx_directory"],
        str(inputs["fx_file_pattern"]),
    )
    index_directory = ROOT / inputs["global_index_directory"]
    indices = {
        str(group["group_id"]): load_global_index(
            index_directory / str(group["index_file"]),
            str(group["index_symbol"]),
        )
        for group in contract["tracked_groups"]
    }
    return panel, master, benchmark, fx, indices


def calendar_successor_maps(
    calendar: pd.DatetimeIndex,
    holding_period: int,
) -> tuple[dict[pd.Timestamp, pd.Timestamp], dict[pd.Timestamp, pd.Timestamp]]:
    """把收盘信号映射到下一开盘和固定持有期退出开盘。"""

    entry_by_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    exit_by_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    for index, signal_date in enumerate(calendar):
        entry_index = index + 1
        exit_index = entry_index + holding_period
        if exit_index >= len(calendar):
            continue
        entry_by_signal[pd.Timestamp(signal_date)] = pd.Timestamp(calendar[entry_index])
        exit_by_signal[pd.Timestamp(signal_date)] = pd.Timestamp(calendar[exit_index])
    return entry_by_signal, exit_by_signal


def _rolling_mad(values: pd.Series, lookback: int) -> pd.Series:
    shifted = values.shift(1)
    return shifted.rolling(lookback, min_periods=lookback).apply(
        lambda array: float(np.median(np.abs(array - np.median(array)))),
        raw=True,
    )


def build_discount_signals(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: pd.DataFrame,
    fx: pd.DataFrame,
    indices: dict[str, pd.DataFrame],
    contract: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """仅用信号日收盘及此前信息构造下一开盘可执行信号。"""

    validate_contract(contract)
    universe = contract["universe"]
    signal_rules = contract["signal"]
    portfolio = contract["portfolio"]
    all_codes = {
        str(code)
        for group in contract["tracked_groups"]
        for code in group["etf_codes"]
    }
    base = panel.loc[
        panel["con_code"].astype(str).isin(all_codes) & panel["date"].le(end)
    ].copy()
    if base.empty:
        raise DataContractError("固定QDII ETF池没有行情")
    metadata = master.rename(columns={"ts_code": "con_code"}).copy()
    allowed_statuses = set(str(value) for value in universe["master_statuses"])
    metadata = metadata.loc[metadata["status"].astype(str).isin(allowed_statuses)]
    base = base.merge(
        metadata[["con_code", "name", "status", "list_date", "delist_date"]],
        on="con_code",
        how="inner",
        validate="many_to_one",
    )
    group_frames: list[pd.DataFrame] = []
    for group_contract in contract["tracked_groups"]:
        group_id = str(group_contract["group_id"])
        codes = set(str(code) for code in group_contract["etf_codes"])
        subset = base.loc[base["con_code"].astype(str).isin(codes)].copy()
        if subset.empty:
            raise DataContractError(f"固定跟踪组没有ETF行情：{group_id}")
        foreign = indices[group_id][["date", "close"]].rename(
            columns={"date": "foreign_date", "close": "foreign_close"}
        )
        fx_column = str(group_contract["fx_series"])
        fx_subset = fx[["date", fx_column]].rename(
            columns={"date": "fx_date", fx_column: "fx_cny_per_foreign"}
        )
        subset["date"] = subset["date"].astype("datetime64[ns]")
        foreign["foreign_date"] = foreign["foreign_date"].astype("datetime64[ns]")
        fx_subset["fx_date"] = fx_subset["fx_date"].astype("datetime64[ns]")
        subset = subset.sort_values("date")
        subset = pd.merge_asof(
            subset,
            foreign.sort_values("foreign_date"),
            left_on="date",
            right_on="foreign_date",
            direction="backward",
            allow_exact_matches=False,
        )
        subset = pd.merge_asof(
            subset.sort_values("date"),
            fx_subset.sort_values("fx_date"),
            left_on="date",
            right_on="fx_date",
            direction="backward",
            allow_exact_matches=True,
        )
        subset["group_id"] = group_id
        subset["index_symbol"] = str(group_contract["index_symbol"])
        subset["fx_series"] = fx_column
        group_frames.append(subset)
    data = pd.concat(group_frames, ignore_index=True)
    data.sort_values(["con_code", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["foreign_age_calendar_days"] = (data["date"] - data["foreign_date"]).dt.days
    data["fx_age_calendar_days"] = (data["date"] - data["fx_date"]).dt.days
    data["relative_log_price"] = (
        np.log(data["total_return_close"].where(data["total_return_close"].gt(0.0)))
        - np.log(data["foreign_close"].where(data["foreign_close"].gt(0.0)))
        - np.log(data["fx_cny_per_foreign"].where(data["fx_cny_per_foreign"].gt(0.0)))
    )
    grouped = data.groupby("con_code", sort=False)
    lookback = int(signal_rules["rolling_equilibrium_lookback_trading_days"])
    data["rolling_relative_median"] = grouped["relative_log_price"].transform(
        lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
    )
    data["rolling_relative_mad"] = grouped["relative_log_price"].transform(
        lambda values: _rolling_mad(values, lookback)
    )
    data["rolling_scale"] = (
        1.4826 * data["rolling_relative_mad"]
    ).clip(lower=float(signal_rules["minimum_scale"]))
    data["relative_log_deviation"] = (
        data["relative_log_price"] - data["rolling_relative_median"]
    )
    data["discount_zscore"] = data["relative_log_deviation"] / data["rolling_scale"]
    amount_lookback = int(universe["amount_lookback_trading_days"])
    data["prior_median_amount_20"] = grouped["amount"].transform(
        lambda values: values.shift(1).rolling(
            amount_lookback,
            min_periods=amount_lookback,
        ).median()
    )
    data["amount_ratio"] = data["amount"] / data["prior_median_amount_20"]
    data["prior_raw_close"] = grouped["raw_close"].shift(1)
    data["log_return_1d"] = grouped["total_return_close"].transform(
        lambda values: np.log(values.where(values.gt(0.0))).diff()
    )
    observed = (
        data["total_return_close"].gt(0.0)
        & data["raw_close"].gt(0.0)
        & data["amount"].gt(0.0)
        & ~data["is_suspended"].fillna(True)
    )
    data["observed_signal_bar"] = observed
    data["observed_history_count"] = observed.astype(int).groupby(data["con_code"]).cumsum()
    calendar = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]).sort_values().unique())
    entry_by_signal, exit_by_signal = calendar_successor_maps(
        calendar,
        int(signal_rules["fixed_holding_period_trading_days_open_to_open"]),
    )
    point_in_time = data["date"].ge(data["list_date"])
    point_in_time &= data["delist_date"].isna() | data["date"].lt(data["delist_date"])
    timing_valid = (
        data["foreign_date"].notna()
        & data["fx_date"].notna()
        & data["foreign_age_calendar_days"].between(
            1,
            int(contract["information_timing"]["maximum_foreign_age_calendar_days"]),
            inclusive="both",
        )
        & data["fx_age_calendar_days"].between(
            0,
            int(contract["information_timing"]["maximum_fx_age_calendar_days"]),
            inclusive="both",
        )
    )
    signal_eligible = (
        data["date"].isin(entry_by_signal)
        & point_in_time
        & timing_valid
        & data["observed_history_count"].ge(int(universe["minimum_observed_history_days"]))
        & data["observed_signal_bar"]
        & data["raw_close"].ge(float(universe["minimum_raw_price_cny"]))
        & data["prior_median_amount_20"].ge(
            float(universe["minimum_prior_median_amount_20_cny"])
        )
        & data["amount_ratio"].ge(
            float(universe["minimum_signal_amount_to_prior_median_ratio"])
        )
        & data["discount_zscore"].le(float(signal_rules["maximum_entry_zscore"]))
        & data["relative_log_deviation"].le(
            float(signal_rules["maximum_entry_log_deviation"])
        )
    )
    signals = data.loc[
        signal_eligible,
        [
            "date",
            "con_code",
            "name",
            "group_id",
            "index_symbol",
            "fx_series",
            "raw_close",
            "discount_zscore",
            "relative_log_deviation",
            "relative_log_price",
            "rolling_relative_median",
            "rolling_scale",
            "foreign_date",
            "fx_date",
            "foreign_age_calendar_days",
            "fx_age_calendar_days",
            "prior_median_amount_20",
            "amount_ratio",
        ],
    ].rename(columns={"date": "signal_date", "raw_close": "signal_raw_close"})
    signals["entry_date"] = signals["signal_date"].map(entry_by_signal)
    signals["scheduled_exit_date"] = signals["signal_date"].map(exit_by_signal)
    signals = signals.loc[
        signals["entry_date"].between(start, end, inclusive="both")
        & signals["scheduled_exit_date"].le(end)
    ].copy()
    entry = data[
        ["date", "con_code", "raw_open", "total_return_open", "amount", "is_suspended"]
    ].rename(
        columns={
            "date": "entry_date",
            "raw_open": "entry_raw_open",
            "total_return_open": "entry_total_return_open",
            "amount": "entry_amount",
            "is_suspended": "entry_is_suspended",
        }
    )
    signals = signals.merge(entry, on=["entry_date", "con_code"], how="left", validate="one_to_one")
    signals["entry_gap"] = signals["entry_raw_open"] / signals["signal_raw_close"] - 1.0
    entry_eligible = (
        signals["entry_raw_open"].ge(float(universe["minimum_raw_price_cny"]))
        & signals["entry_total_return_open"].gt(0.0)
        & signals["entry_amount"].gt(0.0)
        & ~signals["entry_is_suspended"].fillna(True)
        & signals["entry_gap"].gt(float(portfolio["entry_open_gap_lower_bound"]))
        & signals["entry_gap"].lt(float(portfolio["entry_open_gap_upper_bound"]))
    )
    before_entry_filter = int(len(signals))
    signals = signals.loc[entry_eligible].copy()
    signals.sort_values(
        ["signal_date", "group_id", "discount_zscore", "prior_median_amount_20", "con_code"],
        ascending=[True, True, True, False, True],
        inplace=True,
    )
    signals = signals.drop_duplicates(["signal_date", "group_id"], keep="first")
    signals.sort_values(
        ["entry_date", "discount_zscore", "prior_median_amount_20", "con_code"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    signals.reset_index(drop=True, inplace=True)
    market = data[
        [
            "date",
            "con_code",
            "total_return_open",
            "total_return_close",
            "raw_open",
            "raw_close",
            "prior_raw_close",
            "amount",
            "is_suspended",
        ]
    ].copy()
    return_wide = data.pivot(index="date", columns="con_code", values="log_return_1d").sort_index()
    by_year = signals.groupby(signals["signal_date"].dt.year).size()
    audit = {
        "fixed_etf_count": len(all_codes),
        "fixed_tracking_group_count": len(contract["tracked_groups"]),
        "signal_rows_before_entry_filter": before_entry_filter,
        "executable_signal_rows": int(len(signals)),
        "signal_day_count": int(signals["signal_date"].nunique()),
        "entry_day_count": int(signals["entry_date"].nunique()),
        "unique_etf_count": int(signals["con_code"].nunique()),
        "tracking_groups_with_signals": sorted(signals["group_id"].astype(str).unique().tolist()),
        "signal_rows_by_year": {str(int(year)): int(count) for year, count in by_year.items()},
        "first_signal_date": None if signals.empty else signals["signal_date"].min().date().isoformat(),
        "last_signal_date": None if signals.empty else signals["signal_date"].max().date().isoformat(),
        "maximum_foreign_age_calendar_days": None
        if signals.empty
        else int(signals["foreign_age_calendar_days"].max()),
        "maximum_fx_age_calendar_days": None
        if signals.empty
        else int(signals["fx_age_calendar_days"].max()),
        "foreign_date_strictly_before_signal": bool(
            signals.empty or signals["foreign_date"].lt(signals["signal_date"]).all()
        ),
        "fx_date_not_after_signal": bool(
            signals.empty or signals["fx_date"].le(signals["signal_date"]).all()
        ),
        "same_open_used_for_signal_and_execution": False,
        "future_return_used_in_signal_construction": False,
    }
    return signals, market, return_wide, audit


def etf_cost_rate(contract: dict[str, Any], scenario: str) -> float:
    """返回基础或压力情景的单腿全成本率。"""

    costs = contract["costs"]
    slippage = float(costs[f"{scenario}_slippage_bps_per_leg"]) / 10_000.0
    impact = float(costs[f"{scenario}_market_impact_bps_per_leg"]) / 10_000.0
    return (
        float(costs["etf_user_fee_rate_per_leg"])
        + float(costs["conservative_exchange_handling_fee_rate_per_leg"])
        + slippage
        + impact
        + float(costs["stamp_duty_rate"])
    )


def pair_correlation(
    return_wide: pd.DataFrame,
    left: str,
    right: str,
    signal_date: pd.Timestamp,
    *,
    lookback: int,
    minimum_observations: int,
) -> float | None:
    """仅用信号日及此前的配对收益计算相关系数。"""

    if left not in return_wide.columns or right not in return_wide.columns:
        return None
    paired = return_wide.loc[
        return_wide.index <= signal_date,
        [left, right],
    ].tail(lookback).dropna()
    if len(paired) < minimum_observations:
        return None
    value = float(paired[left].corr(paired[right]))
    return value if np.isfinite(value) else None


def run_portfolio_backtest(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    return_wide: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """在空仓时等权进入最多两个跟踪组，并按固定五日开盘退出。"""

    validate_contract(contract)
    account = contract["account"]
    universe = contract["universe"]
    portfolio = contract["portfolio"]
    risk = contract["risk"]
    calendar_frame = benchmark.loc[
        benchmark["date"].between(start, end, inclusive="both")
    ].sort_values("date").reset_index(drop=True)
    if calendar_frame.empty:
        raise DataContractError("评价区间没有H00300交易日")
    market_index = market.set_index(["con_code", "date"]).sort_index()
    if not market_index.index.is_unique:
        raise DataContractError("QDII行情主键重复")
    signals_by_entry = {
        pd.Timestamp(date): group.sort_values(
            ["discount_zscore", "prior_median_amount_20", "con_code"],
            ascending=[True, False, True],
        ).copy()
        for date, group in signals.groupby("entry_date", sort=True)
    } if not signals.empty else {}
    initial_capital = float(account["initial_capital_cny"])
    cash = {"base": initial_capital, "stress": initial_capital}
    prior_nav = {"base": initial_capital, "stress": initial_capital}
    daily_cash_factor = (1.0 + float(account["cash_annual_rate"])) ** (
        1.0 / int(contract["visible_gates"]["annualization_trading_days"])
    )
    positions: list[QdiiPosition] = []
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prior_benchmark_close: float | None = None
    maximum_capacity_fraction = 0.0
    maximum_gross_exposure = 0.0
    maximum_net_exposure = 0.0
    maximum_position_count = 0
    blocked_exit_attempt_count = 0
    stale_mark_count = 0
    correlation_rejection_count = 0
    missing_correlation_rejection_count = 0
    nonflat_entry_skip_count = 0
    entry_candidate_count = 0
    entry_transaction_count = 0
    exit_transaction_count = 0

    def get_market_row(code: str, date: pd.Timestamp) -> pd.Series | None:
        key = (code, date)
        if key not in market_index.index:
            return None
        row = market_index.loc[key]
        if isinstance(row, pd.DataFrame):
            raise DataContractError(f"QDII行情主键不唯一：{code} {date.date()}")
        return row

    def marked_value(
        position: QdiiPosition,
        date: pd.Timestamp,
        field: str,
    ) -> tuple[float, bool]:
        row = get_market_row(position.con_code, date)
        if row is not None:
            value = float(row[field])
            if np.isfinite(value) and value > 0.0:
                return (
                    position.entry_gross_notional
                    * value
                    / position.entry_total_return_open,
                    False,
                )
        return (
            position.entry_gross_notional
            * position.last_total_return_close
            / position.entry_total_return_open,
            True,
        )

    for day_index, benchmark_row in calendar_frame.iterrows():
        date = pd.Timestamp(benchmark_row["date"])
        if day_index > 0:
            cash["base"] *= daily_cash_factor
            cash["stress"] *= daily_cash_factor
        remaining: list[QdiiPosition] = []
        for position in positions:
            if date < position.scheduled_exit_date:
                remaining.append(position)
                continue
            row = get_market_row(position.con_code, date)
            exit_gap: float | None = None
            exit_allowed = False
            if row is not None:
                prior_raw_close = float(row["prior_raw_close"])
                raw_open = float(row["raw_open"])
                total_return_open = float(row["total_return_open"])
                if np.isfinite(prior_raw_close) and prior_raw_close > 0.0:
                    exit_gap = raw_open / prior_raw_close - 1.0
                exit_allowed = bool(
                    not bool(row["is_suspended"])
                    and float(row["amount"]) > 0.0
                    and np.isfinite(raw_open)
                    and raw_open > 0.0
                    and np.isfinite(total_return_open)
                    and total_return_open > 0.0
                    and exit_gap is not None
                    and exit_gap > float(portfolio["exit_open_gap_lower_bound"])
                )
            if not exit_allowed:
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                remaining.append(position)
                continue
            gross, stale = marked_value(position, date, "total_return_open")
            if stale:
                raise DataContractError(
                    f"允许退出但没有有效复权开盘：{position.con_code} {date.date()}"
                )
            for scenario in ("base", "stress"):
                cash[scenario] += gross * (1.0 - etf_cost_rate(contract, scenario))
            exit_transaction_count += 1
            trade_rows.append(
                {
                    "trade_date": date,
                    "side": "SELL",
                    "con_code": position.con_code,
                    "group_id": position.group_id,
                    "shares": position.shares,
                    "gross_notional_cny": gross,
                    "signal_date": position.signal_date,
                    "scheduled_exit_date": position.scheduled_exit_date,
                    "delayed_exit_days": position.delayed_exit_days,
                    "discount_zscore": position.discount_zscore,
                    "relative_log_deviation": position.relative_log_deviation,
                    "capacity_fraction": gross / position.prior_median_amount_20,
                    "open_gap": exit_gap,
                }
            )
        positions = remaining
        candidates = signals_by_entry.get(date, pd.DataFrame())
        if positions and not candidates.empty:
            nonflat_entry_skip_count += int(len(candidates))
        if not positions and not candidates.empty:
            selected_rows: list[Any] = []
            selected_codes: list[str] = []
            selected_groups: set[str] = set()
            for candidate in candidates.itertuples(index=False):
                entry_candidate_count += 1
                code = str(candidate.con_code)
                group_id = str(candidate.group_id)
                if group_id in selected_groups:
                    continue
                acceptable = True
                for peer in selected_codes:
                    correlation = pair_correlation(
                        return_wide,
                        code,
                        peer,
                        pd.Timestamp(candidate.signal_date),
                        lookback=int(portfolio["diversification_correlation_lookback_trading_days"]),
                        minimum_observations=int(
                            portfolio["diversification_minimum_pair_observations"]
                        ),
                    )
                    if correlation is None:
                        missing_correlation_rejection_count += 1
                        acceptable = False
                        break
                    if correlation >= float(portfolio["maximum_pairwise_correlation"]):
                        correlation_rejection_count += 1
                        acceptable = False
                        break
                if not acceptable:
                    continue
                selected_rows.append(candidate)
                selected_codes.append(code)
                selected_groups.add(group_id)
                if len(selected_rows) >= int(portfolio["maximum_positions_per_cycle"]):
                    break
            if selected_rows:
                stress_rate = etf_cost_rate(contract, "stress")
                gross_budget = cash["stress"] / (1.0 + stress_rate)
                target_per_position = gross_budget / len(selected_rows)
                for candidate in selected_rows:
                    code = str(candidate.con_code)
                    row = get_market_row(code, date)
                    if row is None:
                        continue
                    raw_open = float(row["raw_open"])
                    total_return_open = float(row["total_return_open"])
                    board_lot = int(portfolio["board_lot_shares"])
                    shares = int(np.floor(target_per_position / raw_open / board_lot) * board_lot)
                    if shares <= 0:
                        continue
                    gross = shares * raw_open
                    if gross < float(portfolio["minimum_trade_notional_cny"]):
                        continue
                    capacity_fraction = gross / float(candidate.prior_median_amount_20)
                    if capacity_fraction > float(
                        universe["maximum_order_fraction_of_prior_median_amount"]
                    ) + 1e-12:
                        raise DataContractError(f"目标金额超过冻结容量：{code} {date.date()}")
                    if gross * (1.0 + stress_rate) > cash["stress"]:
                        affordable_lots = int(
                            np.floor(
                                cash["stress"]
                                / (raw_open * (1.0 + stress_rate))
                                / board_lot
                            )
                        )
                        shares = affordable_lots * board_lot
                        gross = shares * raw_open
                        capacity_fraction = gross / float(candidate.prior_median_amount_20)
                    if shares <= 0 or gross < float(portfolio["minimum_trade_notional_cny"]):
                        continue
                    for scenario in ("base", "stress"):
                        cash[scenario] -= gross * (1.0 + etf_cost_rate(contract, scenario))
                    if min(cash.values()) < -1e-8:
                        raise DataContractError(f"QDII买入导致现金为负：{code} {date.date()}")
                    positions.append(
                        QdiiPosition(
                            con_code=code,
                            group_id=str(candidate.group_id),
                            shares=shares,
                            entry_date=date,
                            scheduled_exit_date=pd.Timestamp(candidate.scheduled_exit_date),
                            entry_total_return_open=total_return_open,
                            entry_gross_notional=gross,
                            prior_median_amount_20=float(candidate.prior_median_amount_20),
                            signal_date=pd.Timestamp(candidate.signal_date),
                            discount_zscore=float(candidate.discount_zscore),
                            relative_log_deviation=float(candidate.relative_log_deviation),
                            last_total_return_close=total_return_open,
                        )
                    )
                    maximum_capacity_fraction = max(maximum_capacity_fraction, capacity_fraction)
                    entry_transaction_count += 1
                    trade_rows.append(
                        {
                            "trade_date": date,
                            "side": "BUY",
                            "con_code": code,
                            "group_id": str(candidate.group_id),
                            "shares": shares,
                            "gross_notional_cny": gross,
                            "signal_date": pd.Timestamp(candidate.signal_date),
                            "scheduled_exit_date": pd.Timestamp(candidate.scheduled_exit_date),
                            "delayed_exit_days": 0,
                            "discount_zscore": float(candidate.discount_zscore),
                            "relative_log_deviation": float(candidate.relative_log_deviation),
                            "capacity_fraction": capacity_fraction,
                            "open_gap": float(candidate.entry_gap),
                        }
                    )
        close_position_value = 0.0
        for position in positions:
            row = get_market_row(position.con_code, date)
            value, stale = marked_value(position, date, "total_return_close")
            if stale:
                stale_mark_count += 1
            elif row is not None:
                close = float(row["total_return_close"])
                if np.isfinite(close) and close > 0.0:
                    position.last_total_return_close = close
            close_position_value += value
        if date == pd.Timestamp(calendar_frame["date"].iloc[-1]) and positions:
            raise DataContractError(
                f"评价区间末仍有未退出QDII持仓：{sorted(p.con_code for p in positions)}"
            )
        nav = {scenario: cash[scenario] + close_position_value for scenario in ("base", "stress")}
        if min(nav.values()) <= 0.0:
            raise DataContractError(f"QDII组合净值非正：{date.date()}")
        reference_nav = min(nav.values())
        gross_exposure = close_position_value / reference_nav
        net_exposure = gross_exposure
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_net_exposure = max(maximum_net_exposure, abs(net_exposure))
        maximum_position_count = max(maximum_position_count, len(positions))
        if gross_exposure > float(risk["maximum_gross_exposure"]) + 1e-12:
            raise DataContractError(f"毛敞口超过冻结上限：{date.date()} {gross_exposure}")
        if abs(net_exposure) > float(risk["maximum_absolute_net_exposure"]) + 1e-12:
            raise DataContractError(f"净敞口超过冻结上限：{date.date()} {net_exposure}")
        benchmark_close = float(benchmark_row["benchmark_close"])
        benchmark_return = (
            benchmark_close / float(benchmark_row["benchmark_open"]) - 1.0
            if prior_benchmark_close is None
            else benchmark_close / prior_benchmark_close - 1.0
        )
        daily_rows.append(
            {
                "trade_date": date,
                "strategy_base_net_return": nav["base"] / prior_nav["base"] - 1.0,
                "strategy_stress_net_return": nav["stress"] / prior_nav["stress"] - 1.0,
                "benchmark_total_return": benchmark_return,
                "base_nav_cny": nav["base"],
                "stress_nav_cny": nav["stress"],
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "position_count": len(positions),
                "quality_complete": True,
                "capacity_pass": True,
            }
        )
        prior_nav = nav
        prior_benchmark_close = benchmark_close
    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    audit = {
        "initial_capital_cny": initial_capital,
        "final_base_nav_cny": float(daily["base_nav_cny"].iloc[-1]),
        "final_stress_nav_cny": float(daily["stress_nav_cny"].iloc[-1]),
        "entry_candidate_count": int(entry_candidate_count),
        "entry_transaction_count": int(entry_transaction_count),
        "exit_transaction_count": int(exit_transaction_count),
        "nonflat_entry_skip_count": int(nonflat_entry_skip_count),
        "correlation_rejection_count": int(correlation_rejection_count),
        "missing_correlation_rejection_count": int(missing_correlation_rejection_count),
        "blocked_exit_attempt_count": int(blocked_exit_attempt_count),
        "stale_mark_count": int(stale_mark_count),
        "maximum_capacity_fraction": float(maximum_capacity_fraction),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_net_exposure),
        "maximum_position_count": int(maximum_position_count),
        "historical_bid_ask_observed": False,
        "execution_cost_status": "MODELED_BASE_AND_STRESS_COSTS",
    }
    return daily, trades, audit


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    portfolio = report["data_audit"]["portfolio"]
    return "\n".join(
        [
            "# QDII跨市场折价回归V1可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            "## 核心结果",
            "",
            f"- 可见期：{report['period']['start']} 至 {report['period']['end']}。",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}。",
            f"- 压力策略净CAGR：{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础年化净超额：{metrics['base_annualized_excess']:.2%}。",
            f"- 压力年化净超额：{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础净夏普：{metrics['base_strategy_net_sharpe']:.3f}。",
            f"- 压力净夏普：{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 基础最大回撤：{metrics['base_maximum_drawdown']:.2%}。",
            f"- 压力最大回撤：{metrics['stress_maximum_drawdown']:.2%}。",
            f"- 买入/卖出交易数：{portfolio['entry_transaction_count']}/{portfolio['exit_transaction_count']}。",
            "",
            "## 核心硬门",
            "",
            f"- 基础净超额至少40个百分点：{gates['base_annualized_excess_at_least_40pct']}。",
            f"- 压力净超额至少40个百分点：{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础净夏普至少1.50：{gates['base_strategy_sharpe_at_least_1_5']}。",
            f"- 压力净夏普至少1.50：{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部可见期门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            "## 决策",
            "",
            f"{report['decision']['next_step']}。",
            "",
            "封存复验、Paper、Shadow、订单和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """运行一次冻结可见期；不读取封存期绩效。"""

    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, benchmark, fx, indices = load_research_inputs(contract)
    signals, market, return_wide, signal_audit = build_discount_signals(
        panel,
        master,
        benchmark,
        fx,
        indices,
        contract,
        start=start,
        end=end,
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        return_wide,
        contract,
        start=start,
        end=end,
    )
    evaluation = evaluate_historical_returns(
        daily,
        contract,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    status = (
        "VISIBLE_PASS_SEALED_REPLICATION_AUTHORIZED_NOT_OPENED"
        if passed
        else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
    )
    report = {
        "schema_version": "1.0.0",
        "report_id": "QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": {
            "initial_capital_cny": float(contract["account"]["initial_capital_cny"]),
            "user_transaction_fee_rate_per_leg": float(
                contract["account"]["user_transaction_fee_rate_per_leg"]
            ),
            "minimum_annualized_net_excess": float(
                contract["visible_gates"]["minimum_annualized_net_excess"]
            ),
            "minimum_strategy_net_sharpe": float(
                contract["visible_gates"]["minimum_strategy_net_sharpe"]
            ),
        },
        "manifest_verification": manifest_verification,
        "data_audit": {
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "foreign_index_ranges": {
                group_id: {
                    "first_date": frame["date"].min().date().isoformat(),
                    "last_date": frame["date"].max().date().isoformat(),
                    "rows": int(len(frame)),
                }
                for group_id, frame in indices.items()
            },
            "fx_range": {
                "first_date": fx["date"].min().date().isoformat(),
                "last_date": fx["date"].max().date().isoformat(),
                "rows": int(len(fx)),
            },
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "next_step": (
                "可见期全部通过；另行验证清单后才可打开封存复验，当前仍未打开"
                if passed
                else "冻结拒绝本候选，不打开封存期，不调参救回"
            ),
        },
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    output_dir = ROOT / contract["outputs"]["visible_daily_returns"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_dir, index=False)
    signals.to_parquet(ROOT / contract["outputs"]["visible_signals"], index=False)
    trades.to_parquet(ROOT / contract["outputs"]["visible_trades"], index=False)
    atomic_json(ROOT / contract["outputs"]["visible_report_json"], report)
    atomic_text(ROOT / contract["outputs"]["visible_report_markdown"], render_markdown(report))
    return report
