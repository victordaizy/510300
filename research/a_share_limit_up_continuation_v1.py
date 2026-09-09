"""A股涨停后可成交延续V1：冻结信号、组合回测与保守可见期评价。"""

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

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "a_share_limit_up_continuation_v1.yaml"
CONSERVATIVE_EVALUATOR_CONFIG = (
    ROOT / "config" / "qdii_cross_market_discount_reversion_zero_variance_v2.yaml"
)


@dataclass
class StockPosition:
    """基础与压力现金账本共享的一笔A股实物持仓。"""

    con_code: str
    shares: int
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    entry_total_return_open: float
    entry_gross_notional: float
    prior_median_amount_20: float
    signal_date: pd.Timestamp
    signal_limit_fraction: float
    last_total_return_close: float
    delayed_exit_days: int = 0


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取并校验候选合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A股涨停延续合同必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    """阻止目标、成本、时点、组合或安全边界被静默弱化。"""

    failures: list[str] = []
    protocol = contract.get("protocol", {})
    account = contract.get("account", {})
    partition = contract.get("historical_partition", {})
    timing = contract.get("information_timing", {})
    universe = contract.get("universe", {})
    signal = contract.get("signal", {})
    coverage = contract.get("prefreeze_coverage_gate", {})
    portfolio = contract.get("portfolio", {})
    costs = contract.get("costs", {})
    risk = contract.get("risk", {})
    gates = contract.get("visible_gates", {})
    limits = contract.get("historical_evidence_limits", {})
    outputs = contract.get("outputs", {})
    safety = contract.get("safety", {})

    expected_text = {
        "candidate_id": (protocol, "A_SHARE_LIMIT_UP_CONTINUATION_V1"),
        "parent_protocol_id": (
            protocol,
            "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V3",
        ),
        "lane": (protocol, "A_SHARE_PRICE_LIMIT_CONTINUATION"),
        "protocol_status": (protocol, "FROZEN_BEFORE_VISIBLE_OUTCOME"),
        "signal_information_time": (
            timing,
            "SIGNAL_DAY_CLOSE_AFTER_A_SHARE_CLOSING_AUCTION",
        ),
        "first_executable_time": (timing, "NEXT_H00300_TRADING_DAY_OPEN"),
        "required_split_group": (universe, "TRAIN"),
        "signal_family": (
            signal,
            "A_SHARE_CLOSE_AT_UPPER_PRICE_LIMIT_CONTINUATION",
        ),
        "rounding": (signal, "ROUND_HALF_UP_TO_CNY_0_01"),
        "ranking": (
            signal,
            "PRIOR_MEDIAN_AMOUNT_20_DESC_THEN_CODE_ASC",
        ),
        "blocked_exit_policy": (
            portfolio,
            "CARRY_UNTIL_FIRST_ELIGIBLE_OPEN",
        ),
        "terminal_policy": (
            portfolio,
            "FAIL_NO_FORCED_OR_FICTITIOUS_EXIT",
        ),
        "benchmark": (gates, "H00300_TOTAL_RETURN"),
        "bootstrap_method": (gates.get("bootstrap", {}), "PAIRED_MOVING_BLOCK"),
    }
    for name, (section, expected) in expected_text.items():
        key = {
            "protocol_status": "status",
            "signal_family": "family",
            "rounding": "theoretical_limit_price_rounding",
            "ranking": "ranking_when_slots_scarce",
            "terminal_policy": "terminal_open_positions_policy",
            "bootstrap_method": "method",
        }.get(name, name)
        if section.get(key) != expected:
            failures.append(name)

    expected_numbers = {
        "initial_capital_cny": (account, 500_000.0),
        "user_transaction_fee_rate_per_leg": (account, 0.0001),
        "cash_annual_rate": (account, 0.015),
        "minimum_observed_history_days": (universe, 120.0),
        "amount_lookback_trading_days": (universe, 20.0),
        "minimum_prior_median_amount_20_cny": (universe, 200_000_000.0),
        "maximum_order_fraction": (
            universe,
            0.001,
        ),
        "price_tick_cny": (signal, 0.01),
        "limit_price_absolute_tolerance_cny": (signal, 0.0051),
        "main_board_limit_fraction": (signal, 0.10),
        "star_market_limit_fraction": (signal, 0.20),
        "chinext_limit_fraction_before_reform": (signal, 0.10),
        "chinext_limit_fraction_after_reform": (signal, 0.20),
        "fixed_holding_period": (signal, 5.0),
        "minimum_signal_rows": (coverage, 500.0),
        "minimum_signal_years": (coverage, 6.0),
        "minimum_unique_signal_stocks": (coverage, 300.0),
        "maximum_positions": (portfolio, 10.0),
        "target_weight_per_position": (portfolio, 0.10),
        "board_lot_shares": (portfolio, 100.0),
        "minimum_trade_notional_cny": (portfolio, 20_000.0),
        "entry_upper_limit_buffer_fraction": (portfolio, 0.0025),
        "exit_lower_limit_buffer_fraction": (portfolio, 0.0025),
        "stock_user_fee_rate_per_leg": (costs, 0.0001),
        "exchange_and_regulatory_fee": (costs, 0.0000687),
        "base_slippage_bps_per_leg": (costs, 10.0),
        "stress_slippage_bps_per_leg": (costs, 30.0),
        "stamp_duty_before": (costs, 0.001),
        "stamp_duty_after": (costs, 0.0005),
        "maximum_gross_exposure": (risk, 1.0),
        "maximum_absolute_net_exposure": (risk, 1.0),
        "annualization_trading_days": (gates, 242.0),
        "minimum_annualized_net_excess": (gates, 0.40),
        "minimum_strategy_net_sharpe": (gates, 1.50),
        "minimum_rolling_excess_median": (gates, 0.40),
        "minimum_rolling_sharpe_median": (gates, 1.50),
        "bootstrap_repetitions": (gates.get("bootstrap", {}), 5000.0),
    }
    number_keys = {
        "maximum_order_fraction": "maximum_order_fraction_of_prior_median_amount",
        "fixed_holding_period": "fixed_holding_period_trading_days_open_to_open",
        "exchange_and_regulatory_fee": (
            "conservative_exchange_and_regulatory_fee_rate_per_leg"
        ),
        "stamp_duty_before": "stamp_duty_sell_rate_before_reduction",
        "stamp_duty_after": "stamp_duty_sell_rate_after_reduction",
        "bootstrap_repetitions": "repetitions",
    }
    for name, (section, expected) in expected_numbers.items():
        key = number_keys.get(name, name)
        try:
            actual = float(section.get(key, np.nan))
        except (TypeError, ValueError):
            actual = np.nan
        if actual != expected:
            failures.append(name)

    expected_dates = {
        "visible_warmup_start": "2014-01-02",
        "visible_start": "2015-01-05",
        "visible_end": "2022-12-30",
        "formula_family_replication_warmup_start": "2023-01-03",
        "formula_family_replication_start": "2024-01-02",
        "formula_family_replication_end": "2026-08-14",
    }
    for key, expected in expected_dates.items():
        if partition.get(key) != expected:
            failures.append(key)
    if signal.get("chinext_reform_effective_date") != "2020-08-24":
        failures.append("chinext_reform_effective_date")
    if costs.get("stamp_duty_reduction_effective_date") != "2023-08-28":
        failures.append("stamp_duty_reduction_effective_date")

    required_true = {
        "research_only": protocol,
        "replication_may_open_only_after_visible_pass": partition,
        "require_point_in_time_list_and_delist_dates": universe,
        "require_observed_non_suspended_signal_bar": universe,
        "require_observed_non_suspended_entry_bar": universe,
        "require_raw_high_equal_raw_close": signal,
        "fill_available_slots_each_entry_day": portfolio,
        "rank_only_by_liquidity_not_future_outcome": portfolio,
        "stop_new_entries_if_scheduled_exit_after_partition_end": portfolio,
        "omitted_cost_is_failure": costs,
        "base_and_stress_must_both_pass": gates,
        "data_quality_complete_required": gates,
        "capacity_pass_required": gates,
        "replication_requires_exact_daily_limit_input_refresh_before_open": limits,
    }
    for name, section in required_true.items():
        if not bool(section.get(name, False)):
            failures.append(name)

    required_false = {
        "prior_failed_formula_reused": protocol,
        "prior_failed_family_parameter_rescue": protocol,
        "outcome_used_to_choose_parameters": protocol,
        "replication_is_external_blind_holdout": partition,
        "retrospective_history_can_verify_target": partition,
        "signal_day_close_used_for_same_close_execution": timing,
        "next_open_used_in_signal": timing,
        "future_return_used_in_signal": timing,
        "beijing_exchange_allowed": universe,
        "active_position_same_stock_reentry_allowed": signal,
        "parameter_grid_search_allowed": signal,
        "future_open_or_return_may_be_read": coverage,
        "short_sale_allowed": risk,
        "borrowing_allowed": risk,
        "leverage_allowed": risk,
        "derivative_position_allowed": risk,
        "historical_result_may_authorize_paper_or_live": limits,
    }
    for name, section in required_false.items():
        if bool(section.get(name, True)):
            failures.append(name)

    if account.get("user_transaction_fee_rate_per_leg") != costs.get(
        "stock_user_fee_rate_per_leg"
    ):
        failures.append("user_fee_consistency")
    if universe.get("exchanges") != ["SSE", "SZSE"]:
        failures.append("exchanges")
    if universe.get("allowed_hash_buckets") != [1, 2, 3, 4]:
        failures.append("allowed_hash_buckets")
    if universe.get("master_statuses") != ["L", "D"]:
        failures.append("master_statuses")
    if not outputs or len(set(outputs.values())) != len(outputs):
        failures.append("outputs")
    if any(
        bool(safety.get(name, True))
        for name in (
            "paper_position_generation",
            "shadow_signal_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety")
    if failures:
        raise ValueError(f"A股涨停延续合同被弱化或损坏：{sorted(set(failures))}")


def _require_file(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结输入：{relative}")
    return path


def load_visible_inputs(
    contract: dict[str, Any],
    *,
    signal_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """读取可见期输入；冻结覆盖检查时不读取未来开盘或收益列。"""

    validate_contract(contract)
    inputs = contract["inputs"]
    partition = contract["historical_partition"]
    panel_path = _require_file(inputs["visible_panel"])
    master_path = _require_file(inputs["stock_master"])
    benchmark_path = _require_file(inputs["visible_benchmark"])
    start = pd.Timestamp(partition["visible_warmup_start"])
    end = pd.Timestamp(partition["visible_end"])

    signal_columns = [
        "con_code",
        "date",
        "pre_close",
        "raw_high",
        "raw_close",
        "pct_chg",
        "amount",
        "is_suspended",
    ]
    execution_columns = ["raw_open", "total_return_open", "total_return_close"]
    panel_columns = signal_columns if signal_only else signal_columns + execution_columns
    panel = pd.read_parquet(
        panel_path,
        columns=panel_columns,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    master_columns = [
        "ts_code",
        "symbol",
        "exchange",
        "list_status",
        "list_date",
        "delist_date",
        "split_bucket",
        "split_group",
    ]
    master = pd.read_parquet(master_path, columns=master_columns)
    benchmark_columns = ["date"] if signal_only else ["date", "symbol", "close"]
    benchmark = pd.read_parquet(
        benchmark_path,
        columns=benchmark_columns,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )

    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce")
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce")
    if panel.empty or panel["date"].isna().any():
        raise DataContractError("可见A股面板为空或日期无效")
    if panel.duplicated(["con_code", "date"]).any():
        raise DataContractError("可见A股面板股票日期主键重复")
    if benchmark.empty or benchmark["date"].isna().any() or benchmark["date"].duplicated().any():
        raise DataContractError("H00300可见日历为空、日期无效或重复")
    if master["ts_code"].duplicated().any():
        raise DataContractError("股票主表代码重复")

    universe = contract["universe"]
    master = master.loc[
        master["split_group"].eq(universe["required_split_group"])
        & master["split_bucket"].isin(universe["allowed_hash_buckets"])
        & master["exchange"].isin(universe["exchanges"])
        & master["list_status"].isin(universe["master_statuses"])
        & master["list_date"].notna()
    ].copy()
    if master.empty:
        raise DataContractError("冻结训练组没有可用沪深股票")
    allowed_codes = set(master["ts_code"].astype(str))
    raw_panel_rows = int(len(panel))
    panel = panel.loc[panel["con_code"].astype(str).isin(allowed_codes)].copy()
    panel.sort_values(["con_code", "date"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    benchmark.sort_values("date", inplace=True)
    benchmark.reset_index(drop=True, inplace=True)
    if panel.empty:
        raise DataContractError("可见面板与冻结训练组交集为空")
    if not signal_only:
        if not benchmark["symbol"].astype(str).eq("H00300").all():
            raise DataContractError("可见基准并非唯一H00300")
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        if benchmark["close"].isna().any() or (benchmark["close"] <= 0.0).any():
            raise DataContractError("H00300可见收盘价缺失或非正")

    audit = {
        "mode": "SIGNAL_ONLY_PREFREEZE" if signal_only else "FORMAL_VISIBLE",
        "visible_panel_path": panel_path.relative_to(ROOT).as_posix(),
        "stock_master_path": master_path.relative_to(ROOT).as_posix(),
        "visible_benchmark_path": benchmark_path.relative_to(ROOT).as_posix(),
        "panel_columns_read": panel_columns,
        "benchmark_columns_read": benchmark_columns,
        "future_open_or_return_read": bool(not signal_only),
        "raw_panel_rows_in_partition": raw_panel_rows,
        "eligible_panel_rows": int(len(panel)),
        "eligible_master_stocks": int(len(master)),
        "benchmark_calendar_rows": int(len(benchmark)),
        "first_panel_date": panel["date"].min().date().isoformat(),
        "last_panel_date": panel["date"].max().date().isoformat(),
        "first_benchmark_date": benchmark["date"].min().date().isoformat(),
        "last_benchmark_date": benchmark["date"].max().date().isoformat(),
        "sealed_panel_or_benchmark_read": False,
    }
    return panel, master, benchmark, audit


def limit_fraction(con_code: str, date: pd.Timestamp | str) -> float:
    """按代码板块和日期返回冻结的理论涨跌停比例。"""

    symbol = str(con_code).split(".", maxsplit=1)[0]
    timestamp = pd.Timestamp(date)
    if symbol.startswith("688"):
        return 0.20
    if symbol.startswith(("300", "301")):
        return 0.20 if timestamp >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10


def theoretical_upper_limit_price(
    pre_close: pd.Series | np.ndarray | float,
    fractions: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    """按正数价格的人民币分位ROUND_HALF_UP计算理论涨停价。"""

    values = np.asarray(pre_close, dtype=float) * (1.0 + np.asarray(fractions, dtype=float))
    rounded = np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0
    if np.ndim(pre_close) == 0 and np.ndim(fractions) == 0:
        return float(rounded)
    if isinstance(pre_close, pd.Series):
        return pd.Series(rounded, index=pre_close.index, dtype=float)
    return rounded


def build_limit_up_signals(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只用信号日及此前字段构造信号与未来交易日历日期。"""

    validate_contract(contract)
    required_panel = {
        "con_code",
        "date",
        "pre_close",
        "raw_high",
        "raw_close",
        "pct_chg",
        "amount",
        "is_suspended",
    }
    required_master = {
        "ts_code",
        "list_date",
        "delist_date",
        "split_bucket",
        "split_group",
        "exchange",
        "list_status",
    }
    missing_panel = required_panel.difference(panel.columns)
    missing_master = required_master.difference(master.columns)
    if missing_panel or missing_master or "date" not in benchmark.columns:
        raise DataContractError(
            f"涨停信号输入缺字段：panel={sorted(missing_panel)} "
            f"master={sorted(missing_master)} benchmark_date={'date' in benchmark.columns}"
        )
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    data = panel[list(required_panel)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data.sort_values(["con_code", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    if data.empty or data["date"].isna().any() or data.duplicated(["con_code", "date"]).any():
        raise DataContractError("涨停信号面板为空、日期无效或主键重复")

    for column in ("pre_close", "raw_high", "raw_close", "pct_chg", "amount"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["observed_history_days"] = data.groupby("con_code", sort=False).cumcount() + 1
    lookback = int(contract["universe"]["amount_lookback_trading_days"])
    data["prior_median_amount_20"] = data.groupby(
        "con_code", sort=False, group_keys=False
    )["amount"].transform(
        lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
    )

    master_fields = master[
        [
            "ts_code",
            "list_date",
            "delist_date",
            "split_bucket",
            "split_group",
            "exchange",
            "list_status",
        ]
    ].rename(columns={"ts_code": "con_code"})
    data = data.merge(master_fields, on="con_code", how="inner", validate="many_to_one")
    data["list_date"] = pd.to_datetime(data["list_date"], errors="coerce")
    data["delist_date"] = pd.to_datetime(data["delist_date"], errors="coerce")
    if data.empty:
        raise DataContractError("涨停信号面板与主表合并后为空")

    symbols = data["con_code"].astype(str).str.split(".", n=1).str[0]
    dates = data["date"]
    fractions = np.full(len(data), float(contract["signal"]["main_board_limit_fraction"]))
    fractions[symbols.str.startswith("688").to_numpy()] = float(
        contract["signal"]["star_market_limit_fraction"]
    )
    chinext = symbols.str.startswith(("300", "301")).to_numpy()
    after_reform = dates.ge(pd.Timestamp(contract["signal"]["chinext_reform_effective_date"])).to_numpy()
    fractions[chinext & after_reform] = float(
        contract["signal"]["chinext_limit_fraction_after_reform"]
    )
    fractions[chinext & ~after_reform] = float(
        contract["signal"]["chinext_limit_fraction_before_reform"]
    )
    data["limit_fraction"] = fractions
    data["theoretical_upper_limit_price"] = theoretical_upper_limit_price(
        data["pre_close"], data["limit_fraction"]
    )

    universe = contract["universe"]
    signal_config = contract["signal"]
    tolerance = float(signal_config["limit_price_absolute_tolerance_cny"])
    finite_required = np.isfinite(
        data[
            [
                "pre_close",
                "raw_high",
                "raw_close",
                "pct_chg",
                "amount",
                "prior_median_amount_20",
                "theoretical_upper_limit_price",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    point_in_time_listed = data["date"].ge(data["list_date"]) & (
        data["delist_date"].isna() | data["date"].le(data["delist_date"])
    )
    observed_bar = ~data["is_suspended"].fillna(True).astype(bool) & data["amount"].gt(0.0)
    history_pass = data["observed_history_days"].ge(
        int(universe["minimum_observed_history_days"])
    )
    liquidity_pass = data["prior_median_amount_20"].ge(
        float(universe["minimum_prior_median_amount_20_cny"])
    )
    price_pass = data["raw_close"].ge(float(universe["minimum_signal_raw_close_cny"]))
    close_at_limit = data["raw_close"].sub(data["theoretical_upper_limit_price"]).abs().le(tolerance)
    high_equals_close = data["raw_high"].sub(data["raw_close"]).abs().le(tolerance)
    observed_move_pass = data["pct_chg"].div(100.0).ge(
        data["limit_fraction"]
        * float(signal_config["minimum_observed_fraction_of_theoretical_limit_move"])
    )
    date_pass = data["date"].between(start, end, inclusive="both")
    candidate_mask = (
        finite_required
        & point_in_time_listed.to_numpy()
        & observed_bar.to_numpy()
        & history_pass.to_numpy()
        & liquidity_pass.to_numpy()
        & price_pass.to_numpy()
        & close_at_limit.to_numpy()
        & high_equals_close.to_numpy()
        & observed_move_pass.to_numpy()
        & date_pass.to_numpy()
    )
    candidates = data.loc[candidate_mask].copy()

    calendar = pd.DatetimeIndex(
        pd.to_datetime(benchmark["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    )
    if calendar.empty:
        raise DataContractError("H00300日历为空")
    calendar_position = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    candidates["calendar_position"] = candidates["date"].map(calendar_position)
    holding = int(signal_config["fixed_holding_period_trading_days_open_to_open"])
    candidates = candidates.loc[candidates["calendar_position"].notna()].copy()
    candidates["entry_calendar_position"] = candidates["calendar_position"].astype(int) + 1
    candidates["exit_calendar_position"] = candidates["entry_calendar_position"] + holding
    candidates = candidates.loc[candidates["exit_calendar_position"].lt(len(calendar))].copy()
    candidates["entry_date"] = calendar.take(
        candidates["entry_calendar_position"].astype(int).to_numpy()
    )
    candidates["scheduled_exit_date"] = calendar.take(
        candidates["exit_calendar_position"].astype(int).to_numpy()
    )
    candidates = candidates.loc[candidates["scheduled_exit_date"].le(end)].copy()
    candidates.rename(columns={"date": "signal_date"}, inplace=True)
    output_columns = [
        "con_code",
        "signal_date",
        "entry_date",
        "scheduled_exit_date",
        "prior_median_amount_20",
        "observed_history_days",
        "pre_close",
        "raw_high",
        "raw_close",
        "pct_chg",
        "limit_fraction",
        "theoretical_upper_limit_price",
        "split_bucket",
        "split_group",
        "exchange",
    ]
    signals = candidates[output_columns].copy()
    signals.sort_values(
        ["entry_date", "prior_median_amount_20", "con_code"],
        ascending=[True, False, True],
        inplace=True,
    )
    signals.reset_index(drop=True, inplace=True)
    years = sorted(int(value) for value in signals["signal_date"].dt.year.unique())
    audit = {
        "signal_rows": int(len(signals)),
        "unique_signal_stocks": int(signals["con_code"].nunique()),
        "signal_years": years,
        "signal_year_count": int(len(years)),
        "first_signal_date": (
            signals["signal_date"].min().date().isoformat() if not signals.empty else None
        ),
        "last_signal_date": (
            signals["signal_date"].max().date().isoformat() if not signals.empty else None
        ),
        "input_panel_rows": int(len(panel)),
        "master_stock_count": int(master["ts_code"].nunique()),
        "same_close_execution_used": False,
        "future_open_or_return_read": False,
        "future_price_or_return_columns_used": [],
        "calendar_dates_only_used_after_signal": True,
        "parameter_grid_search_used": False,
    }
    return signals, audit


def stock_cost_rate(
    contract: dict[str, Any],
    *,
    scenario: str,
    side: str,
    date: pd.Timestamp | str,
) -> float:
    """返回指定情景、买卖方向和日期的冻结单腿全成本率。"""

    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知成本情景：{scenario}")
    normalized_side = str(side).upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise ValueError(f"未知交易方向：{side}")
    costs = contract["costs"]
    rate = (
        float(costs["stock_user_fee_rate_per_leg"])
        + float(costs["conservative_exchange_and_regulatory_fee_rate_per_leg"])
        + float(costs[f"{scenario}_slippage_bps_per_leg"]) / 10_000.0
    )
    if normalized_side == "SELL":
        effective = pd.Timestamp(costs["stamp_duty_reduction_effective_date"])
        rate += float(
            costs[
                "stamp_duty_sell_rate_after_reduction"
                if pd.Timestamp(date) >= effective
                else "stamp_duty_sell_rate_before_reduction"
            ]
        )
    return float(rate)


def run_portfolio_backtest(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按下一开盘可成交、十只上限和五个开盘周期运行双成本账本。"""

    validate_contract(contract)
    required_market = {
        "con_code",
        "date",
        "pre_close",
        "raw_open",
        "amount",
        "is_suspended",
        "total_return_open",
        "total_return_close",
    }
    required_signals = {
        "con_code",
        "signal_date",
        "entry_date",
        "scheduled_exit_date",
        "prior_median_amount_20",
        "limit_fraction",
    }
    missing_market = required_market.difference(market.columns)
    missing_signals = required_signals.difference(signals.columns)
    if missing_market or missing_signals or not {"date", "close"}.issubset(benchmark.columns):
        raise DataContractError(
            f"回测输入缺字段：market={sorted(missing_market)} "
            f"signals={sorted(missing_signals)} benchmark={sorted({'date', 'close'}.difference(benchmark.columns))}"
        )
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    market_data = market[list(required_market)].copy()
    market_data["date"] = pd.to_datetime(market_data["date"], errors="coerce")
    if market_data["date"].isna().any() or market_data.duplicated(["con_code", "date"]).any():
        raise DataContractError("回测行情日期无效或股票日期主键重复")
    market_index = market_data.set_index(["con_code", "date"]).sort_index()

    benchmark_data = benchmark[["date", "close"]].copy()
    benchmark_data["date"] = pd.to_datetime(benchmark_data["date"], errors="coerce")
    benchmark_data["close"] = pd.to_numeric(benchmark_data["close"], errors="coerce")
    benchmark_data.sort_values("date", inplace=True)
    if (
        benchmark_data.empty
        or benchmark_data["date"].isna().any()
        or benchmark_data["date"].duplicated().any()
        or benchmark_data["close"].isna().any()
        or (benchmark_data["close"] <= 0.0).any()
    ):
        raise DataContractError("H00300基准为空、重复、缺失或非正")
    benchmark_data["benchmark_total_return"] = benchmark_data["close"].pct_change()
    calendar = benchmark_data.loc[
        benchmark_data["date"].between(start, end, inclusive="both")
    ].copy()
    if calendar.empty or calendar["benchmark_total_return"].isna().any():
        raise DataContractError("评价区间缺少H00300前收或有效日收益")
    calendar.reset_index(drop=True, inplace=True)

    signal_data = signals.copy()
    for column in ("signal_date", "entry_date", "scheduled_exit_date"):
        signal_data[column] = pd.to_datetime(signal_data[column], errors="coerce")
    if signal_data[list(("signal_date", "entry_date", "scheduled_exit_date"))].isna().any().any():
        raise DataContractError("信号执行日期无效")
    signals_by_entry = {
        pd.Timestamp(date): group.sort_values(
            ["prior_median_amount_20", "con_code"], ascending=[False, True]
        ).copy()
        for date, group in signal_data.groupby("entry_date", sort=True)
    } if not signal_data.empty else {}

    account = contract["account"]
    universe = contract["universe"]
    portfolio = contract["portfolio"]
    risk = contract["risk"]
    initial_capital = float(account["initial_capital_cny"])
    cash = {"base": initial_capital, "stress": initial_capital}
    prior_nav = {"base": initial_capital, "stress": initial_capital}
    trading_days = int(contract["visible_gates"]["annualization_trading_days"])
    daily_cash_factor = (1.0 + float(account["cash_annual_rate"])) ** (1.0 / trading_days)
    positions: list[StockPosition] = []
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    entry_candidate_count = 0
    entry_transaction_count = 0
    exit_transaction_count = 0
    active_reentry_skip_count = 0
    invalid_entry_bar_skip_count = 0
    upper_limit_entry_skip_count = 0
    minimum_notional_skip_count = 0
    capacity_limited_entry_count = 0
    blocked_exit_attempt_count = 0
    stale_mark_count = 0
    maximum_capacity_fraction = 0.0
    maximum_gross_exposure = 0.0
    maximum_net_exposure = 0.0
    maximum_position_count = 0

    def get_market_row(code: str, date: pd.Timestamp) -> pd.Series | None:
        key = (code, date)
        if key not in market_index.index:
            return None
        row = market_index.loc[key]
        if isinstance(row, pd.DataFrame):
            raise DataContractError(f"行情主键不唯一：{code} {date.date()}")
        return row

    def marked_value(
        position: StockPosition,
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

    for benchmark_row in calendar.itertuples(index=False):
        date = pd.Timestamp(benchmark_row.date)
        for scenario in ("base", "stress"):
            cash[scenario] *= daily_cash_factor

        remaining: list[StockPosition] = []
        for position in positions:
            if date < position.scheduled_exit_date:
                remaining.append(position)
                continue
            row = get_market_row(position.con_code, date)
            exit_gap: float | None = None
            exit_allowed = False
            if row is not None:
                pre_close = float(row["pre_close"])
                raw_open = float(row["raw_open"])
                total_return_open = float(row["total_return_open"])
                amount = float(row["amount"])
                if np.isfinite(pre_close) and pre_close > 0.0 and np.isfinite(raw_open):
                    exit_gap = raw_open / pre_close - 1.0
                lower_bound = -limit_fraction(position.con_code, date) + float(
                    portfolio["exit_lower_limit_buffer_fraction"]
                )
                exit_allowed = bool(
                    not bool(row["is_suspended"])
                    and np.isfinite(amount)
                    and amount > 0.0
                    and np.isfinite(raw_open)
                    and raw_open > 0.0
                    and np.isfinite(total_return_open)
                    and total_return_open > 0.0
                    and exit_gap is not None
                    and exit_gap > lower_bound
                )
            if not exit_allowed:
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                remaining.append(position)
                continue
            gross, stale = marked_value(position, date, "total_return_open")
            if stale:
                raise DataContractError(
                    f"允许退出但缺少有效复权开盘：{position.con_code} {date.date()}"
                )
            base_rate = stock_cost_rate(
                contract, scenario="base", side="SELL", date=date
            )
            stress_rate = stock_cost_rate(
                contract, scenario="stress", side="SELL", date=date
            )
            cash["base"] += gross * (1.0 - base_rate)
            cash["stress"] += gross * (1.0 - stress_rate)
            exit_transaction_count += 1
            trade_rows.append(
                {
                    "trade_date": date,
                    "side": "SELL",
                    "con_code": position.con_code,
                    "shares": position.shares,
                    "gross_notional_cny": gross,
                    "base_cost_rate": base_rate,
                    "stress_cost_rate": stress_rate,
                    "base_cash_flow_cny": gross * (1.0 - base_rate),
                    "stress_cash_flow_cny": gross * (1.0 - stress_rate),
                    "signal_date": position.signal_date,
                    "entry_date": position.entry_date,
                    "scheduled_exit_date": position.scheduled_exit_date,
                    "delayed_exit_days": position.delayed_exit_days,
                    "capacity_fraction": gross / position.prior_median_amount_20,
                    "open_gap": exit_gap,
                }
            )
        positions = remaining

        candidates = signals_by_entry.get(date, pd.DataFrame())
        if not candidates.empty and len(positions) < int(portfolio["maximum_positions"]):
            active_codes = {position.con_code for position in positions}
            for candidate in candidates.itertuples(index=False):
                if len(positions) >= int(portfolio["maximum_positions"]):
                    break
                entry_candidate_count += 1
                code = str(candidate.con_code)
                if code in active_codes:
                    active_reentry_skip_count += 1
                    continue
                if pd.Timestamp(candidate.scheduled_exit_date) > end:
                    continue
                row = get_market_row(code, date)
                valid_bar = False
                entry_gap: float | None = None
                if row is not None:
                    pre_close = float(row["pre_close"])
                    raw_open = float(row["raw_open"])
                    total_return_open = float(row["total_return_open"])
                    amount = float(row["amount"])
                    if np.isfinite(pre_close) and pre_close > 0.0 and np.isfinite(raw_open):
                        entry_gap = raw_open / pre_close - 1.0
                    valid_bar = bool(
                        not bool(row["is_suspended"])
                        and np.isfinite(amount)
                        and amount > 0.0
                        and np.isfinite(raw_open)
                        and raw_open > 0.0
                        and np.isfinite(total_return_open)
                        and total_return_open > 0.0
                    )
                if not valid_bar or row is None or entry_gap is None:
                    invalid_entry_bar_skip_count += 1
                    continue
                upper_bound = limit_fraction(code, date) - float(
                    portfolio["entry_upper_limit_buffer_fraction"]
                )
                if entry_gap >= upper_bound:
                    upper_limit_entry_skip_count += 1
                    continue

                raw_open = float(row["raw_open"])
                target_notional = min(prior_nav.values()) * float(
                    portfolio["target_weight_per_position"]
                )
                capacity_notional = float(candidate.prior_median_amount_20) * float(
                    universe["maximum_order_fraction_of_prior_median_amount"]
                )
                if capacity_notional + 1e-12 < target_notional:
                    capacity_limited_entry_count += 1
                stress_buy_rate = stock_cost_rate(
                    contract, scenario="stress", side="BUY", date=date
                )
                available_notional = cash["stress"] / (1.0 + stress_buy_rate)
                gross_budget = min(target_notional, capacity_notional, available_notional)
                board_lot = int(portfolio["board_lot_shares"])
                shares = int(np.floor(gross_budget / raw_open / board_lot) * board_lot)
                gross = shares * raw_open
                if shares <= 0 or gross < float(portfolio["minimum_trade_notional_cny"]):
                    minimum_notional_skip_count += 1
                    continue
                capacity_fraction = gross / float(candidate.prior_median_amount_20)
                if capacity_fraction > float(
                    universe["maximum_order_fraction_of_prior_median_amount"]
                ) + 1e-12:
                    raise DataContractError(f"订单容量越界：{code} {date.date()}")
                base_rate = stock_cost_rate(
                    contract, scenario="base", side="BUY", date=date
                )
                stress_rate = stress_buy_rate
                cash["base"] -= gross * (1.0 + base_rate)
                cash["stress"] -= gross * (1.0 + stress_rate)
                if min(cash.values()) < -1e-8:
                    raise DataContractError(f"买入导致现金为负：{code} {date.date()}")
                total_return_open = float(row["total_return_open"])
                position = StockPosition(
                    con_code=code,
                    shares=shares,
                    entry_date=date,
                    scheduled_exit_date=pd.Timestamp(candidate.scheduled_exit_date),
                    entry_total_return_open=total_return_open,
                    entry_gross_notional=gross,
                    prior_median_amount_20=float(candidate.prior_median_amount_20),
                    signal_date=pd.Timestamp(candidate.signal_date),
                    signal_limit_fraction=float(candidate.limit_fraction),
                    last_total_return_close=total_return_open,
                )
                positions.append(position)
                active_codes.add(code)
                entry_transaction_count += 1
                maximum_capacity_fraction = max(maximum_capacity_fraction, capacity_fraction)
                trade_rows.append(
                    {
                        "trade_date": date,
                        "side": "BUY",
                        "con_code": code,
                        "shares": shares,
                        "gross_notional_cny": gross,
                        "base_cost_rate": base_rate,
                        "stress_cost_rate": stress_rate,
                        "base_cash_flow_cny": -gross * (1.0 + base_rate),
                        "stress_cash_flow_cny": -gross * (1.0 + stress_rate),
                        "signal_date": pd.Timestamp(candidate.signal_date),
                        "entry_date": date,
                        "scheduled_exit_date": pd.Timestamp(candidate.scheduled_exit_date),
                        "delayed_exit_days": 0,
                        "capacity_fraction": capacity_fraction,
                        "open_gap": entry_gap,
                    }
                )

        close_position_value = 0.0
        for position in positions:
            value, stale = marked_value(position, date, "total_return_close")
            if stale:
                stale_mark_count += 1
            else:
                row = get_market_row(position.con_code, date)
                if row is not None:
                    current_close = float(row["total_return_close"])
                    if np.isfinite(current_close) and current_close > 0.0:
                        position.last_total_return_close = current_close
            close_position_value += value

        if date == pd.Timestamp(calendar["date"].iloc[-1]) and positions:
            open_codes = sorted(position.con_code for position in positions)
            raise DataContractError(f"评价期末仍有未退出持仓：{open_codes}")
        nav = {
            scenario: cash[scenario] + close_position_value
            for scenario in ("base", "stress")
        }
        if min(nav.values()) <= 0.0 or not all(np.isfinite(value) for value in nav.values()):
            raise DataContractError(f"组合净值无效：{date.date()}")
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
        daily_rows.append(
            {
                "trade_date": date,
                "strategy_base_net_return": nav["base"] / prior_nav["base"] - 1.0,
                "strategy_stress_net_return": nav["stress"] / prior_nav["stress"] - 1.0,
                "benchmark_total_return": float(benchmark_row.benchmark_total_return),
                "base_nav_cny": nav["base"],
                "stress_nav_cny": nav["stress"],
                "cash_base_cny": cash["base"],
                "cash_stress_cny": cash["stress"],
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "position_count": len(positions),
                "quality_complete": True,
                "capacity_pass": True,
            }
        )
        prior_nav = nav

    daily = pd.DataFrame(daily_rows)
    trade_columns = [
        "trade_date",
        "side",
        "con_code",
        "shares",
        "gross_notional_cny",
        "base_cost_rate",
        "stress_cost_rate",
        "base_cash_flow_cny",
        "stress_cash_flow_cny",
        "signal_date",
        "entry_date",
        "scheduled_exit_date",
        "delayed_exit_days",
        "capacity_fraction",
        "open_gap",
    ]
    trades = pd.DataFrame(trade_rows, columns=trade_columns)
    audit = {
        "initial_capital_cny": initial_capital,
        "final_base_nav_cny": float(daily["base_nav_cny"].iloc[-1]),
        "final_stress_nav_cny": float(daily["stress_nav_cny"].iloc[-1]),
        "entry_candidate_count": int(entry_candidate_count),
        "entry_transaction_count": int(entry_transaction_count),
        "exit_transaction_count": int(exit_transaction_count),
        "active_reentry_skip_count": int(active_reentry_skip_count),
        "invalid_entry_bar_skip_count": int(invalid_entry_bar_skip_count),
        "upper_limit_entry_skip_count": int(upper_limit_entry_skip_count),
        "minimum_notional_skip_count": int(minimum_notional_skip_count),
        "capacity_limited_entry_count": int(capacity_limited_entry_count),
        "blocked_exit_attempt_count": int(blocked_exit_attempt_count),
        "stale_mark_count": int(stale_mark_count),
        "maximum_capacity_fraction": float(maximum_capacity_fraction),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_net_exposure),
        "maximum_position_count": int(maximum_position_count),
        "terminal_open_position_count": 0,
        "same_close_execution_used": False,
        "historical_order_queue_observed": False,
        "execution_cost_status": "MODELED_BASE_AND_STRESS_ALL_IN_STOCK_COSTS",
    }
    return daily, trades, audit


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """以严格JSON原子写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """原子写入UTF-8文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _format_float(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    """生成决策导向的可见期Markdown报告。"""

    evaluation = report.get("evaluation", {})
    metrics = evaluation.get("metrics", {})
    portfolio = report.get("data_audit", {}).get("portfolio", {})
    lines = [
        "# A股涨停后可成交延续V1可见期结果",
        "",
        f"状态：`{report['status']}`",
        "",
        "本结果仅是公式族可见历史筛选，不能验证未来40个百分点超额，也不授权Paper、Shadow、订单或实盘。",
        "",
        "## 核心结果",
        "",
        f"- H00300全收益年化：{_format_percent(metrics.get('benchmark_total_return_cagr'))}",
        f"- 基础/压力策略净年化：{_format_percent(metrics.get('strategy_base_net_cagr'))} / {_format_percent(metrics.get('strategy_stress_net_cagr'))}",
        f"- 基础/压力年化净超额：{_format_percent(metrics.get('base_annualized_excess'))} / {_format_percent(metrics.get('stress_annualized_excess'))}",
        f"- 基础/压力净夏普：{_format_float(metrics.get('base_strategy_net_sharpe'))} / {_format_float(metrics.get('stress_strategy_net_sharpe'))}",
        f"- 基础/压力最大回撤：{_format_percent(metrics.get('base_maximum_drawdown'))} / {_format_percent(metrics.get('stress_maximum_drawdown'))}",
        f"- 基础/压力期末净值：{portfolio.get('final_base_nav_cny', 0.0):,.2f}元 / {portfolio.get('final_stress_nav_cny', 0.0):,.2f}元",
        f"- 买入/卖出笔数：{portfolio.get('entry_transaction_count', 0)} / {portfolio.get('exit_transaction_count', 0)}",
        "",
        "## 硬门",
        "",
    ]
    for name, passed in evaluation.get("gates", {}).items():
        lines.append(f"- `{name}`：{'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 决策",
            "",
            report["decision"]["next_step"],
            "",
            "封存复验未读取；Paper、Shadow、订单、券商连接和实盘均关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """运行一次冻结的可见期路径并写出不可覆盖的研究产物。"""

    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, benchmark, input_audit = load_visible_inputs(
        contract, signal_only=False
    )
    signals, signal_audit = build_limit_up_signals(
        panel,
        master,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        signals,
        panel,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    correction_contract = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        contract,
        correction_contract,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    evaluation["evaluation_semantics"] = (
        "FROZEN_VISIBLE_HISTORICAL_DISCOVERY_WITH_CONSERVATIVE_ZERO_VARIANCE"
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    status = (
        "VISIBLE_PASS_REPLICATION_BLOCKED_PENDING_EXACT_DAILY_LIMIT_INPUT"
        if passed
        else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
    )
    objective = {
        "initial_capital_cny": float(contract["account"]["initial_capital_cny"]),
        "user_transaction_fee_rate_per_leg": float(
            contract["account"]["user_transaction_fee_rate_per_leg"]
        ),
        "benchmark": contract["visible_gates"]["benchmark"],
        "minimum_annualized_net_excess": float(
            contract["visible_gates"]["minimum_annualized_net_excess"]
        ),
        "minimum_strategy_net_sharpe": float(
            contract["visible_gates"]["minimum_strategy_net_sharpe"]
        ),
        "base_and_stress_must_both_pass": True,
    }
    report = {
        "schema_version": "1.0.0",
        "report_id": "A_SHARE_LIMIT_UP_CONTINUATION_V1_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "performance_metrics_available": True,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": objective,
        "manifest_verification": manifest_verification,
        "data_audit": {
            "inputs": input_audit,
            "signal": signal_audit,
            "portfolio": portfolio_audit,
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "visible_economic_and_statistical_gates_pass": passed,
            "formula_family_replication_conditionally_eligible": passed,
            "exact_daily_limit_input_refresh_required_before_replication": passed,
            "sealed_replication_authorized": False,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "next_step": (
                "可见期硬门全部通过；先另起版本获取并冻结精确逐日涨跌停输入，再决定是否打开非外部盲样本的公式族复验"
                if passed
                else "冻结拒绝本候选；不打开2024—2026复验，不改变本候选参数救回，继续登记新的独立机制"
            ),
        },
        "outputs": contract["outputs"],
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_signals"], signals)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
