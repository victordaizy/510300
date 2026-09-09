"""中国可转债传统双低周频轮动V1：冻结信号、成交与保守历史评价。"""

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
CONFIG = ROOT / "config" / "cb_double_low_rotation_v1.yaml"
CONSERVATIVE_EVALUATOR_CONFIG = (
    ROOT / "config" / "qdii_cross_market_discount_reversion_zero_variance_v2.yaml"
)


@dataclass
class CbPosition:
    """基础与压力账本共享的一笔可转债实物持仓。"""

    bond_code: str
    units: int
    entry_date: pd.Timestamp
    entry_open: float
    entry_gross_notional: float
    signal_date: pd.Timestamp
    prior_median_turnover_20: float
    double_low_score: float
    last_close: float
    pending_exit: bool = False
    exit_signal_date: pd.Timestamp | None = None
    delayed_exit_days: int = 0


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取并校验候选合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("可转债双低合同必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    """阻止目标、成本、时点、持仓和安全边界被静默弱化。"""

    protocol = contract.get("protocol", {})
    account = contract.get("account", {})
    inputs = contract.get("inputs", {})
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
    failures: list[str] = []

    expected_text = {
        "candidate_id": (protocol, "CB_DOUBLE_LOW_ROTATION_V1"),
        "parent_protocol_id": (
            protocol,
            "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V4",
        ),
        "lane": (protocol, "CHINA_CONVERTIBLE_BOND_RELATIVE_VALUE"),
        "protocol_status": (protocol, "FROZEN_BEFORE_VISIBLE_OUTCOME"),
        "signal_information_time": (
            timing,
            "LAST_H00300_TRADING_DAY_OF_ISO_WEEK_CLOSE",
        ),
        "first_executable_time": (timing, "NEXT_H00300_TRADING_DAY_OPEN"),
        "signal_family": (signal, "TRADITIONAL_CONVERTIBLE_BOND_DOUBLE_LOW"),
        "ranking": (
            signal,
            "DOUBLE_LOW_SCORE_ASC_THEN_PRIOR_MEDIAN_TURNOVER_DESC_THEN_BOND_CODE_ASC",
        ),
        "pending_exit_policy": (
            signal,
            "PENDING_EXIT_REMAINS_MANDATORY",
        ),
        "target_notional_basis": (
            portfolio,
            "MINIMUM_BASE_STRESS_PRIOR_CLOSE_NAV",
        ),
        "blocked_entry_policy": (
            portfolio,
            "SKIP_AND_HOLD_CASH_UNTIL_NEXT_FROZEN_REBALANCE",
        ),
        "blocked_exit_policy": (portfolio, "CARRY_UNTIL_FIRST_ELIGIBLE_OPEN"),
        "terminal_policy": (
            portfolio,
            "FAIL_NO_FORCED_OR_FICTITIOUS_REDEMPTION",
        ),
        "benchmark": (gates, "H00300_TOTAL_RETURN"),
        "bootstrap_method": (gates.get("bootstrap", {}), "PAIRED_MOVING_BLOCK"),
        "performance_label": (
            limits,
            "DISCOVERY_ONLY_CB_FORMULA_FAMILY_VISIBLE",
        ),
    }
    text_key_overrides = {
        "protocol_status": "status",
        "signal_family": "family",
        "ranking": "ranking",
        "pending_exit_policy": "blocked_exit_reselection_policy",
        "terminal_policy": "terminal_open_positions_policy",
        "bootstrap_method": "method",
        "performance_label": "performance_screen_label",
    }
    for name, (section, expected) in expected_text.items():
        key = text_key_overrides.get(name, name)
        if section.get(key) != expected:
            failures.append(name)

    expected_paths = {
        "source_status": "data/raw/cb_double_low_v1/collection_status.json",
        "bond_master": "data/raw/cb_double_low_v1/bond_master.parquet",
        "visible_panel": "data/raw/cb_double_low_v1/visible_panel_through_2022.parquet",
        "sealed_replication_panel": "data/raw/cb_double_low_v1/sealed_replication_panel_2023_2026.parquet",
        "visible_benchmark": "data/raw/a_share_hash_holdout_v2/training_H00300.parquet",
        "sealed_replication_benchmark": "data/raw/a_share_bucket1_time_holdout_formula_v1/H00300.parquet",
        "parent_manifest": "config/multi_asset_annual_excess_40pct_high_sharpe_v4_manifest.json",
        "conservative_sharpe_evaluator": "research/qdii_cross_market_discount_reversion_zero_variance_v2.py",
    }
    for key, expected in expected_paths.items():
        if inputs.get(key) != expected:
            failures.append(f"input_{key}")

    expected_numbers = {
        "initial_capital_cny": (account, 500_000.0),
        "user_transaction_fee_rate_per_leg": (account, 0.0001),
        "cash_annual_rate": (account, 0.015),
        "minimum_observed_history_days": (universe, 60.0),
        "amount_lookback_trading_days": (universe, 20.0),
        "minimum_prior_median_turnover_notional_cny": (universe, 50_000_000.0),
        "maximum_order_fraction": (universe, 0.001),
        "minimum_signal_close_cny": (universe, 90.0),
        "maximum_signal_close_cny": (universe, 125.0),
        "minimum_conversion_premium_rate_pct": (universe, 0.0),
        "maximum_conversion_premium_rate_pct": (universe, 40.0),
        "maximum_cross_source_close_difference_cny": (universe, 0.01),
        "target_bond_count": (signal, 10.0),
        "minimum_selection_dates": (coverage, 150.0),
        "minimum_eligible_bond_date_rows": (coverage, 500.0),
        "minimum_unique_eligible_bonds": (coverage, 80.0),
        "minimum_signal_years": (coverage, 5.0),
        "maximum_positions": (portfolio, 10.0),
        "target_weight_per_position": (portfolio, 0.10),
        "board_lot_bonds": (portfolio, 10.0),
        "minimum_trade_notional_cny": (portfolio, 10_000.0),
        "maximum_entry_open_gap_fraction": (portfolio, 0.19),
        "minimum_exit_open_gap_fraction": (portfolio, -0.19),
        "cb_user_fee_rate_per_leg": (costs, 0.0001),
        "conservative_exchange_handling_fee_rate_per_leg": (costs, 0.00004),
        "base_slippage_bps_per_leg": (costs, 10.0),
        "stress_slippage_bps_per_leg": (costs, 30.0),
        "base_market_impact_bps_per_leg": (costs, 5.0),
        "stress_market_impact_bps_per_leg": (costs, 15.0),
        "stamp_duty_rate": (costs, 0.0),
        "maximum_gross_exposure": (risk, 1.0),
        "maximum_absolute_net_exposure": (risk, 1.0),
        "annualization_trading_days": (gates, 242.0),
        "minimum_annualized_net_excess": (gates, 0.40),
        "minimum_strategy_net_sharpe": (gates, 1.50),
        "minimum_rolling_excess_median": (gates, 0.40),
        "minimum_rolling_sharpe_median": (gates, 1.50),
        "bootstrap_repetitions": (gates.get("bootstrap", {}), 5000.0),
    }
    number_key_overrides = {
        "maximum_order_fraction": "maximum_order_fraction_of_prior_median_turnover",
        "bootstrap_repetitions": "repetitions",
    }
    for name, (section, expected) in expected_numbers.items():
        key = number_key_overrides.get(name, name)
        try:
            actual = float(section.get(key, np.nan))
        except (TypeError, ValueError):
            actual = np.nan
        if actual != expected:
            failures.append(name)

    expected_dates = {
        "visible_warmup_start": "2017-01-03",
        "visible_start": "2018-01-02",
        "visible_end": "2022-12-30",
        "formula_family_replication_warmup_start": "2023-01-03",
        "formula_family_replication_start": "2023-04-03",
        "formula_family_replication_end": "2026-08-14",
    }
    for key, expected in expected_dates.items():
        if partition.get(key) != expected:
            failures.append(key)

    required_true = {
        "research_only": protocol,
        "replication_may_open_only_after_visible_pass": partition,
        "include_delisted_bonds": universe,
        "require_observed_positive_volume_signal_bar": universe,
        "require_observed_positive_volume_entry_bar": universe,
        "require_point_in_time_conversion_premium": universe,
        "retain_existing_position_if_still_selected": signal,
        "sell_position_if_no_longer_selected": signal,
        "fill_available_slots_each_execution_day": portfolio,
        "stop_new_entries_on_evaluation_end": portfolio,
        "omitted_cost_is_failure": costs,
        "base_and_stress_must_both_pass": gates,
        "data_quality_complete_required": gates,
        "capacity_pass_required": gates,
        "delisted_bonds_included": limits,
        "point_in_time_daily_conversion_premium_available": limits,
        "next_open_fill_is_modeled_from_daily_bar_not_order_queue": limits,
        "disappearance_or_terminal_exit_without_observed_open_is_failure": limits,
    }
    for name, section in required_true.items():
        if not bool(section.get(name, False)):
            failures.append(name)

    required_false = {
        "prior_failed_formula_reused": protocol,
        "prior_cb_priority_allocation_family_reused": protocol,
        "outcome_used_to_choose_parameters": protocol,
        "replication_is_external_blind_holdout": partition,
        "retrospective_history_can_verify_target": partition,
        "same_close_execution_used": timing,
        "next_open_used_in_signal": timing,
        "future_return_used_in_signal": timing,
        "current_survivor_list_only_allowed": universe,
        "parameter_grid_search_allowed": signal,
        "future_open_or_return_may_be_read": coverage,
        "short_sale_allowed": risk,
        "borrowing_allowed": risk,
        "leverage_allowed": risk,
        "stock_conversion_allowed": risk,
        "derivative_position_allowed": risk,
        "immutable_vendor_vintage_chain_available": limits,
        "point_in_time_strong_redemption_announcement_history_used": limits,
        "point_in_time_credit_rating_history_used": limits,
        "historical_intraday_order_queue_available": limits,
        "historical_result_may_authorize_paper_or_live": limits,
    }
    for name, section in required_false.items():
        if bool(section.get(name, True)):
            failures.append(name)

    if universe.get("exchanges") != ["SSE", "SZSE"]:
        failures.append("exchanges")
    if account.get("user_transaction_fee_rate_per_leg") != costs.get(
        "cb_user_fee_rate_per_leg"
    ):
        failures.append("user_fee_consistency")
    if int(signal.get("target_bond_count", 0)) != int(
        portfolio.get("maximum_positions", -1)
    ):
        failures.append("target_position_consistency")
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
        raise ValueError(f"可转债双低合同被弱化或损坏：{sorted(set(failures))}")


def _require_file(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结可见输入：{relative}")
    return path


def _normalise_bond_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def load_visible_inputs(
    contract: dict[str, Any],
    *,
    signal_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """读取可见期输入；冻结覆盖检查不读取开盘或任何收益列。"""

    validate_contract(contract)
    inputs = contract["inputs"]
    partition = contract["historical_partition"]
    status_path = _require_file(inputs["source_status"])
    master_path = _require_file(inputs["bond_master"])
    panel_path = _require_file(inputs["visible_panel"])
    benchmark_path = _require_file(inputs["visible_benchmark"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    source_checks = {
        "collection_complete": status.get("status") == "COLLECTION_COMPLETE",
        "failed_bond_count_zero": int(status.get("failed_bond_count", -1)) == 0,
        "all_requested_completed": int(status.get("requested_bond_count", -1))
        == int(status.get("completed_bond_count", -2)),
        "strategy_return_or_rank_not_computed": status.get(
            "strategy_return_or_rank_computed"
        )
        is False,
        "physical_partition_no_overlap": int(
            status.get("physical_partition_overlap_rows", -1)
        )
        == 0,
        "physical_partition_date_order_valid": status.get(
            "physical_partition_date_order_valid"
        )
        is True,
        "visible_path_matches": status.get("visible_panel_path")
        == inputs["visible_panel"],
        "master_path_matches": status.get("master_path") == inputs["bond_master"],
    }
    if not all(source_checks.values()):
        raise DataContractError(f"可转债采集状态不合格：{source_checks}")

    start = pd.Timestamp(partition["visible_warmup_start"])
    end = pd.Timestamp(partition["visible_end"])
    signal_columns = [
        "bond_code",
        "date",
        "close",
        "volume",
        "turnover_notional_proxy_cny",
        "conversion_premium_rate_pct",
        "value_table_close",
        "close_cross_source_absolute_difference",
    ]
    execution_columns = ["open"]
    panel_columns = signal_columns if signal_only else signal_columns + execution_columns
    panel = pd.read_parquet(
        panel_path,
        columns=panel_columns,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    master_columns = ["bond_code", "listing_date", "exchange", "source"]
    master = pd.read_parquet(master_path, columns=master_columns)
    benchmark_columns = ["date"] if signal_only else ["date", "symbol", "close"]
    benchmark = pd.read_parquet(
        benchmark_path,
        columns=benchmark_columns,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )

    panel["bond_code"] = _normalise_bond_code(panel["bond_code"])
    master["bond_code"] = _normalise_bond_code(master["bond_code"])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    if panel.empty or panel["date"].isna().any():
        raise DataContractError("可见可转债面板为空或日期无效")
    if panel.duplicated(["bond_code", "date"]).any():
        raise DataContractError("可见可转债面板存在债券日期重复")
    if master.empty or master["bond_code"].duplicated().any():
        raise DataContractError("可转债清单为空或代码重复")
    if master["listing_date"].isna().any():
        raise DataContractError("可转债清单存在无效上市日期")
    if not set(master["exchange"].dropna().astype(str).unique()).issubset(
        {"SSE", "SZSE"}
    ):
        raise DataContractError("可转债清单混入非沪深交易所证券")
    if not master["bond_code"].str.startswith(("11", "12")).all():
        raise DataContractError("可转债清单混入非沪深可转债代码")
    if benchmark.empty or benchmark["date"].isna().any() or benchmark["date"].duplicated().any():
        raise DataContractError("H00300日历为空、日期无效或重复")

    numeric_panel = [column for column in panel_columns if column not in {"bond_code", "date"}]
    for column in numeric_panel:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if not signal_only:
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        if benchmark["close"].isna().any() or (benchmark["close"] <= 0.0).any():
            raise DataContractError("H00300可见收盘价缺失或非正")
        if not benchmark["symbol"].astype(str).eq("H00300").all():
            raise DataContractError("可见基准不是H00300全收益指数")

    panel.sort_values(["bond_code", "date"], inplace=True)
    master.sort_values("bond_code", inplace=True)
    benchmark.sort_values("date", inplace=True)
    panel.reset_index(drop=True, inplace=True)
    master.reset_index(drop=True, inplace=True)
    benchmark.reset_index(drop=True, inplace=True)
    panel_codes = set(panel["bond_code"].unique())
    master_codes = set(master["bond_code"].unique())
    unknown_codes = sorted(panel_codes.difference(master_codes))
    if unknown_codes:
        raise DataContractError(f"可见面板存在清单外债券：{unknown_codes[:10]}")

    audit = {
        "source_checks": source_checks,
        "source_status_path": inputs["source_status"],
        "bond_master_path": inputs["bond_master"],
        "visible_panel_path": inputs["visible_panel"],
        "visible_benchmark_path": inputs["visible_benchmark"],
        "panel_row_count": int(len(panel)),
        "panel_bond_count": int(panel["bond_code"].nunique()),
        "master_bond_count": int(len(master)),
        "benchmark_day_count": int(len(benchmark)),
        "first_panel_date": panel["date"].min().date().isoformat(),
        "last_panel_date": panel["date"].max().date().isoformat(),
        "panel_columns_read": panel_columns,
        "benchmark_columns_read": benchmark_columns,
        "execution_open_read": bool(not signal_only),
        "future_open_or_return_read": bool(not signal_only),
        "sealed_panel_or_benchmark_read": False,
        "physical_partition_overlap_rows": 0,
        "strategy_return_or_rank_computed_during_acquisition": False,
        "visible_panel_sha256_from_collection_status": status.get(
            "visible_panel_sha256"
        ),
        "sealed_replication_panel_sha256_from_status_only": status.get(
            "sealed_replication_panel_sha256"
        ),
    }
    return panel, master, benchmark, audit


def _weekly_schedule(
    benchmark: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """生成每个ISO周最后交易日及其下一交易日，不读取价格。"""

    if "date" not in benchmark.columns:
        raise DataContractError("周频日历缺少date")
    dates = pd.DataFrame(
        {"date": pd.to_datetime(benchmark["date"], errors="coerce")}
    ).drop_duplicates()
    dates.sort_values("date", inplace=True)
    if dates.empty or dates["date"].isna().any():
        raise DataContractError("周频日历为空或日期无效")
    dates["next_date"] = dates["date"].shift(-1)
    iso = dates["date"].dt.isocalendar()
    dates["iso_year"] = iso["year"].astype(int)
    dates["iso_week"] = iso["week"].astype(int)
    last_rows = dates.groupby(["iso_year", "iso_week"], sort=True).tail(1).copy()
    schedule = last_rows.rename(
        columns={"date": "signal_date", "next_date": "execution_date"}
    )[["signal_date", "execution_date", "iso_year", "iso_week"]]
    schedule = schedule.loc[
        schedule["execution_date"].notna()
        & schedule["execution_date"].between(
            pd.Timestamp(start), pd.Timestamp(end), inclusive="both"
        )
    ].copy()
    schedule.sort_values("execution_date", inplace=True)
    schedule.reset_index(drop=True, inplace=True)
    if schedule["execution_date"].duplicated().any():
        raise DataContractError("周频执行日重复")
    return schedule


def build_weekly_selections(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """仅用当日收盘和此前成交额构建冻结双低选择。"""

    validate_contract(contract)
    required_panel = {
        "bond_code",
        "date",
        "close",
        "volume",
        "turnover_notional_proxy_cny",
        "conversion_premium_rate_pct",
        "value_table_close",
        "close_cross_source_absolute_difference",
    }
    required_master = {"bond_code", "listing_date", "exchange"}
    missing_panel = required_panel.difference(panel.columns)
    missing_master = required_master.difference(master.columns)
    if missing_panel or missing_master or "date" not in benchmark.columns:
        raise DataContractError(
            f"信号输入缺字段：panel={sorted(missing_panel)} master={sorted(missing_master)}"
        )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    columns = sorted(required_panel)
    data = panel[columns].copy()
    data["bond_code"] = _normalise_bond_code(data["bond_code"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in required_panel.difference({"bond_code", "date"}):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.empty or data["date"].isna().any() or data.duplicated(["bond_code", "date"]).any():
        raise DataContractError("双低信号面板为空、日期无效或主键重复")
    data.sort_values(["bond_code", "date"], inplace=True)
    data["observed_history_days"] = data.groupby("bond_code").cumcount() + 1
    lookback = int(contract["universe"]["amount_lookback_trading_days"])
    data["prior_median_turnover_20"] = data.groupby(
        "bond_code", sort=False
    )["turnover_notional_proxy_cny"].transform(
        lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
    )

    master_data = master[list(required_master)].copy()
    master_data["bond_code"] = _normalise_bond_code(master_data["bond_code"])
    master_data["listing_date"] = pd.to_datetime(
        master_data["listing_date"], errors="coerce"
    )
    if master_data["bond_code"].duplicated().any():
        raise DataContractError("双低信号清单代码重复")
    data = data.merge(master_data, on="bond_code", how="left", validate="many_to_one")
    if data["exchange"].isna().any():
        raise DataContractError("双低信号面板存在清单外债券")

    schedule = _weekly_schedule(benchmark, start=start_ts, end=end_ts)
    signal_dates = schedule[["signal_date", "execution_date"]]
    weekly = data.merge(
        signal_dates,
        left_on="date",
        right_on="signal_date",
        how="inner",
        validate="many_to_one",
    )
    universe = contract["universe"]
    finite_columns = [
        "close",
        "volume",
        "prior_median_turnover_20",
        "conversion_premium_rate_pct",
        "value_table_close",
        "close_cross_source_absolute_difference",
    ]
    finite_mask = pd.Series(True, index=weekly.index)
    if not weekly.empty:
        finite_mask = pd.Series(
            np.isfinite(weekly[finite_columns].to_numpy(dtype=float)).all(axis=1),
            index=weekly.index,
        )
    eligible_mask = (
        finite_mask
        & weekly["exchange"].isin(universe["exchanges"])
        & weekly["observed_history_days"].ge(
            int(universe["minimum_observed_history_days"])
        )
        & weekly["volume"].gt(0.0)
        & weekly["prior_median_turnover_20"].ge(
            float(universe["minimum_prior_median_turnover_notional_cny"])
        )
        & weekly["close"].between(
            float(universe["minimum_signal_close_cny"]),
            float(universe["maximum_signal_close_cny"]),
            inclusive="both",
        )
        & weekly["conversion_premium_rate_pct"].between(
            float(universe["minimum_conversion_premium_rate_pct"]),
            float(universe["maximum_conversion_premium_rate_pct"]),
            inclusive="both",
        )
        & weekly["close_cross_source_absolute_difference"].le(
            float(universe["maximum_cross_source_close_difference_cny"])
        )
    )
    eligible = weekly.loc[eligible_mask].copy()
    eligible["double_low_score"] = (
        eligible["close"] + eligible["conversion_premium_rate_pct"]
    )
    eligible.sort_values(
        [
            "signal_date",
            "double_low_score",
            "prior_median_turnover_20",
            "bond_code",
        ],
        ascending=[True, True, False, True],
        inplace=True,
    )
    if not eligible.empty:
        eligible["eligible_count_on_signal_date"] = eligible.groupby(
            "signal_date"
        )["bond_code"].transform("size")
        eligible["selection_rank"] = eligible.groupby("signal_date").cumcount() + 1
    else:
        eligible["eligible_count_on_signal_date"] = pd.Series(dtype="int64")
        eligible["selection_rank"] = pd.Series(dtype="int64")
    target_count = int(contract["signal"]["target_bond_count"])
    selected = eligible.loc[eligible["selection_rank"].le(target_count)].copy()
    selected_columns = [
        "signal_date",
        "execution_date",
        "bond_code",
        "exchange",
        "selection_rank",
        "eligible_count_on_signal_date",
        "double_low_score",
        "close",
        "conversion_premium_rate_pct",
        "prior_median_turnover_20",
        "observed_history_days",
        "volume",
        "close_cross_source_absolute_difference",
    ]
    selected = selected.reindex(columns=selected_columns)
    selected.sort_values(["execution_date", "selection_rank", "bond_code"], inplace=True)
    selected.reset_index(drop=True, inplace=True)

    selected_dates = selected["signal_date"].drop_duplicates().sort_values()
    eligible_dates = eligible["signal_date"].drop_duplicates().sort_values()
    signal_years = sorted(int(year) for year in selected_dates.dt.year.unique())
    full_dates = int(
        selected.groupby("signal_date")["bond_code"].size().eq(target_count).sum()
    ) if not selected.empty else 0
    audit = {
        "weekly_schedule_count": int(len(schedule)),
        "selection_date_count": int(len(selected_dates)),
        "full_target_selection_date_count": full_dates,
        "under_target_selection_date_count": int(len(selected_dates) - full_dates),
        "empty_target_schedule_count": int(len(schedule) - len(eligible_dates)),
        "eligible_bond_date_rows": int(len(eligible)),
        "selected_bond_date_rows": int(len(selected)),
        "unique_eligible_bonds": int(eligible["bond_code"].nunique()),
        "unique_selected_bonds": int(selected["bond_code"].nunique()),
        "signal_year_count": int(len(signal_years)),
        "signal_years": signal_years,
        "first_signal_date": (
            selected_dates.iloc[0].date().isoformat() if len(selected_dates) else None
        ),
        "last_signal_date": (
            selected_dates.iloc[-1].date().isoformat() if len(selected_dates) else None
        ),
        "first_execution_date": (
            selected["execution_date"].min().date().isoformat()
            if not selected.empty
            else None
        ),
        "last_execution_date": (
            selected["execution_date"].max().date().isoformat()
            if not selected.empty
            else None
        ),
        "double_low_score_formula": contract["signal"]["double_low_score_formula"],
        "future_price_or_return_columns_used": [],
        "future_open_or_return_read": False,
        "sealed_replication_read": False,
    }
    return selected, audit


def cb_cost_rate(
    contract: dict[str, Any],
    *,
    scenario: str,
    side: str,
) -> float:
    """返回冻结的可转债单边全成本率。"""

    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知成本情景：{scenario}")
    if side.upper() not in {"BUY", "SELL"}:
        raise ValueError(f"未知交易方向：{side}")
    costs = contract["costs"]
    rate = (
        float(costs["cb_user_fee_rate_per_leg"])
        + float(costs["conservative_exchange_handling_fee_rate_per_leg"])
        + float(costs[f"{scenario}_slippage_bps_per_leg"]) / 10_000.0
        + float(costs[f"{scenario}_market_impact_bps_per_leg"]) / 10_000.0
    )
    if float(costs["stamp_duty_rate"]) != 0.0:
        raise DataContractError("可转债冻结成本不允许股票印花税")
    return float(rate)


def run_portfolio_backtest(
    selections: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按下一开盘、周频目标集合和双成本现金账本运行组合。"""

    validate_contract(contract)
    required_market = {"bond_code", "date", "open", "close", "volume"}
    required_selections = {
        "signal_date",
        "execution_date",
        "bond_code",
        "selection_rank",
        "double_low_score",
        "prior_median_turnover_20",
    }
    missing_market = required_market.difference(market.columns)
    missing_selections = required_selections.difference(selections.columns)
    missing_benchmark = {"date", "close"}.difference(benchmark.columns)
    if missing_market or missing_selections or missing_benchmark:
        raise DataContractError(
            f"回测输入缺字段：market={sorted(missing_market)} "
            f"selections={sorted(missing_selections)} benchmark={sorted(missing_benchmark)}"
        )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise DataContractError("评价开始日晚于结束日")

    market_data = market[list(required_market)].copy()
    market_data["bond_code"] = _normalise_bond_code(market_data["bond_code"])
    market_data["date"] = pd.to_datetime(market_data["date"], errors="coerce")
    for column in ("open", "close", "volume"):
        market_data[column] = pd.to_numeric(market_data[column], errors="coerce")
    if market_data.empty or market_data["date"].isna().any():
        raise DataContractError("回测可转债行情为空或日期无效")
    if market_data.duplicated(["bond_code", "date"]).any():
        raise DataContractError("回测可转债行情主键重复")
    market_data.sort_values(["bond_code", "date"], inplace=True)
    market_data["prior_close"] = market_data.groupby("bond_code")["close"].shift(1)
    market_index = market_data.set_index(["bond_code", "date"]).sort_index()

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
        benchmark_data["date"].between(start_ts, end_ts, inclusive="both")
    ].copy()
    if calendar.empty or calendar["benchmark_total_return"].isna().any():
        raise DataContractError("评价区间缺少H00300前收或有效日收益")
    calendar.reset_index(drop=True, inplace=True)
    schedule = _weekly_schedule(benchmark_data, start=start_ts, end=end_ts)
    schedule_map = {
        pd.Timestamp(row.execution_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
    }

    selection_data = selections.copy()
    selection_data["bond_code"] = _normalise_bond_code(selection_data["bond_code"])
    for column in ("signal_date", "execution_date"):
        selection_data[column] = pd.to_datetime(selection_data[column], errors="coerce")
    if selection_data[["signal_date", "execution_date"]].isna().any().any():
        raise DataContractError("双低选择日期无效")
    if selection_data.duplicated(["execution_date", "bond_code"]).any():
        raise DataContractError("同一执行日重复选择同一债券")
    if not selection_data.empty and not selection_data["execution_date"].isin(
        list(schedule_map)
    ).all():
        raise DataContractError("双低选择执行日不属于冻结周频日历")
    selections_by_execution = {
        pd.Timestamp(date): group.sort_values(
            ["selection_rank", "bond_code"], ascending=[True, True]
        ).copy()
        for date, group in selection_data.groupby("execution_date", sort=True)
    }

    account = contract["account"]
    universe = contract["universe"]
    portfolio = contract["portfolio"]
    risk = contract["risk"]
    initial_capital = float(account["initial_capital_cny"])
    cash = {"base": initial_capital, "stress": initial_capital}
    prior_nav = {"base": initial_capital, "stress": initial_capital}
    trading_days = int(contract["visible_gates"]["annualization_trading_days"])
    daily_cash_factor = (1.0 + float(account["cash_annual_rate"])) ** (
        1.0 / trading_days
    )
    positions: dict[str, CbPosition] = {}
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    rebalance_count = 0
    empty_target_rebalance_count = 0
    retained_position_count = 0
    pending_exit_reselected_count = 0
    entry_candidate_count = 0
    entry_transaction_count = 0
    exit_transaction_count = 0
    invalid_entry_bar_skip_count = 0
    upper_gap_entry_skip_count = 0
    minimum_notional_skip_count = 0
    capacity_limited_entry_count = 0
    maximum_position_limit_skip_count = 0
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
            raise DataContractError(f"可转债行情主键不唯一：{code} {date.date()}")
        return row

    final_date = pd.Timestamp(calendar["date"].iloc[-1])
    for benchmark_row in calendar.itertuples(index=False):
        date = pd.Timestamp(benchmark_row.date)
        for scenario in ("base", "stress"):
            cash[scenario] *= daily_cash_factor

        target_group = selections_by_execution.get(date)
        is_rebalance = date in schedule_map
        if is_rebalance:
            rebalance_count += 1
            target_codes = (
                set(target_group["bond_code"].astype(str))
                if target_group is not None and not target_group.empty
                else set()
            )
            if not target_codes:
                empty_target_rebalance_count += 1
            for position in positions.values():
                if position.pending_exit:
                    if position.bond_code in target_codes:
                        pending_exit_reselected_count += 1
                    continue
                if position.bond_code in target_codes:
                    retained_position_count += 1
                else:
                    position.pending_exit = True
                    position.exit_signal_date = schedule_map[date]

        is_terminal = date == final_date
        if is_terminal:
            for position in positions.values():
                if not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = date

        for code in sorted(list(positions)):
            position = positions[code]
            if not position.pending_exit:
                continue
            row = get_market_row(code, date)
            exit_gap: float | None = None
            exit_allowed = False
            if row is not None:
                open_price = float(row["open"])
                prior_close = float(row["prior_close"])
                volume = float(row["volume"])
                if (
                    np.isfinite(open_price)
                    and open_price > 0.0
                    and np.isfinite(prior_close)
                    and prior_close > 0.0
                ):
                    exit_gap = open_price / prior_close - 1.0
                exit_allowed = bool(
                    np.isfinite(volume)
                    and volume > 0.0
                    and exit_gap is not None
                    and exit_gap
                    > float(portfolio["minimum_exit_open_gap_fraction"])
                )
            if not exit_allowed or row is None or exit_gap is None:
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                continue
            open_price = float(row["open"])
            gross = position.units * open_price
            base_rate = cb_cost_rate(contract, scenario="base", side="SELL")
            stress_rate = cb_cost_rate(contract, scenario="stress", side="SELL")
            cash["base"] += gross * (1.0 - base_rate)
            cash["stress"] += gross * (1.0 - stress_rate)
            exit_transaction_count += 1
            trade_rows.append(
                {
                    "trade_date": date,
                    "side": "SELL",
                    "bond_code": code,
                    "units": position.units,
                    "gross_notional_cny": gross,
                    "base_cost_rate": base_rate,
                    "stress_cost_rate": stress_rate,
                    "base_cash_flow_cny": gross * (1.0 - base_rate),
                    "stress_cash_flow_cny": gross * (1.0 - stress_rate),
                    "signal_date": position.signal_date,
                    "exit_signal_date": position.exit_signal_date,
                    "entry_date": position.entry_date,
                    "delayed_exit_days": position.delayed_exit_days,
                    "capacity_fraction_at_entry": position.entry_gross_notional
                    / position.prior_median_turnover_20,
                    "open_gap": exit_gap,
                    "double_low_score": position.double_low_score,
                }
            )
            del positions[code]

        if is_rebalance and not is_terminal and target_group is not None:
            active_codes = set(positions)
            for candidate in target_group.itertuples(index=False):
                entry_candidate_count += 1
                code = str(candidate.bond_code)
                if code in active_codes:
                    continue
                if len(positions) >= int(portfolio["maximum_positions"]):
                    maximum_position_limit_skip_count += 1
                    continue
                row = get_market_row(code, date)
                entry_gap: float | None = None
                valid_bar = False
                if row is not None:
                    open_price = float(row["open"])
                    prior_close = float(row["prior_close"])
                    volume = float(row["volume"])
                    if (
                        np.isfinite(open_price)
                        and open_price > 0.0
                        and np.isfinite(prior_close)
                        and prior_close > 0.0
                    ):
                        entry_gap = open_price / prior_close - 1.0
                    valid_bar = bool(
                        np.isfinite(volume)
                        and volume > 0.0
                        and entry_gap is not None
                    )
                if not valid_bar or row is None or entry_gap is None:
                    invalid_entry_bar_skip_count += 1
                    continue
                if entry_gap >= float(portfolio["maximum_entry_open_gap_fraction"]):
                    upper_gap_entry_skip_count += 1
                    continue
                open_price = float(row["open"])
                target_notional = min(prior_nav.values()) * float(
                    portfolio["target_weight_per_position"]
                )
                prior_median = float(candidate.prior_median_turnover_20)
                capacity_notional = prior_median * float(
                    universe["maximum_order_fraction_of_prior_median_turnover"]
                )
                if capacity_notional + 1e-12 < target_notional:
                    capacity_limited_entry_count += 1
                stress_rate = cb_cost_rate(contract, scenario="stress", side="BUY")
                available_notional = min(cash.values()) / (1.0 + stress_rate)
                gross_budget = min(target_notional, capacity_notional, available_notional)
                board_lot = int(portfolio["board_lot_bonds"])
                units = int(np.floor(gross_budget / open_price / board_lot) * board_lot)
                gross = units * open_price
                if units <= 0 or gross < float(portfolio["minimum_trade_notional_cny"]):
                    minimum_notional_skip_count += 1
                    continue
                capacity_fraction = gross / prior_median
                if capacity_fraction > float(
                    universe["maximum_order_fraction_of_prior_median_turnover"]
                ) + 1e-12:
                    raise DataContractError(f"可转债订单容量越界：{code} {date.date()}")
                base_rate = cb_cost_rate(contract, scenario="base", side="BUY")
                cash["base"] -= gross * (1.0 + base_rate)
                cash["stress"] -= gross * (1.0 + stress_rate)
                if min(cash.values()) < -1e-8:
                    raise DataContractError(f"可转债买入导致现金为负：{code} {date.date()}")
                position = CbPosition(
                    bond_code=code,
                    units=units,
                    entry_date=date,
                    entry_open=open_price,
                    entry_gross_notional=gross,
                    signal_date=pd.Timestamp(candidate.signal_date),
                    prior_median_turnover_20=prior_median,
                    double_low_score=float(candidate.double_low_score),
                    last_close=open_price,
                )
                positions[code] = position
                active_codes.add(code)
                entry_transaction_count += 1
                maximum_capacity_fraction = max(
                    maximum_capacity_fraction, capacity_fraction
                )
                trade_rows.append(
                    {
                        "trade_date": date,
                        "side": "BUY",
                        "bond_code": code,
                        "units": units,
                        "gross_notional_cny": gross,
                        "base_cost_rate": base_rate,
                        "stress_cost_rate": stress_rate,
                        "base_cash_flow_cny": -gross * (1.0 + base_rate),
                        "stress_cash_flow_cny": -gross * (1.0 + stress_rate),
                        "signal_date": pd.Timestamp(candidate.signal_date),
                        "exit_signal_date": pd.NaT,
                        "entry_date": date,
                        "delayed_exit_days": 0,
                        "capacity_fraction_at_entry": capacity_fraction,
                        "open_gap": entry_gap,
                        "double_low_score": float(candidate.double_low_score),
                    }
                )

        close_position_value = 0.0
        for position in positions.values():
            row = get_market_row(position.bond_code, date)
            close_price = np.nan if row is None else float(row["close"])
            if np.isfinite(close_price) and close_price > 0.0:
                position.last_close = close_price
            else:
                stale_mark_count += 1
                close_price = position.last_close
            close_position_value += position.units * close_price

        if is_terminal and positions:
            open_codes = sorted(positions)
            raise DataContractError(f"评价期末仍有无法退出的可转债持仓：{open_codes}")
        nav = {
            scenario: cash[scenario] + close_position_value
            for scenario in ("base", "stress")
        }
        if min(nav.values()) <= 0.0 or not all(
            np.isfinite(value) for value in nav.values()
        ):
            raise DataContractError(f"可转债组合净值无效：{date.date()}")
        reference_nav = min(nav.values())
        gross_exposure = close_position_value / reference_nav
        net_exposure = gross_exposure
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_net_exposure = max(maximum_net_exposure, abs(net_exposure))
        maximum_position_count = max(maximum_position_count, len(positions))
        if gross_exposure > float(risk["maximum_gross_exposure"]) + 1e-12:
            raise DataContractError(f"可转债毛敞口超过冻结上限：{date.date()}")
        if abs(net_exposure) > float(risk["maximum_absolute_net_exposure"]) + 1e-12:
            raise DataContractError(f"可转债净敞口超过冻结上限：{date.date()}")
        daily_rows.append(
            {
                "trade_date": date,
                "strategy_base_net_return": nav["base"] / prior_nav["base"] - 1.0,
                "strategy_stress_net_return": nav["stress"] / prior_nav["stress"]
                - 1.0,
                "benchmark_total_return": float(benchmark_row.benchmark_total_return),
                "base_nav_cny": nav["base"],
                "stress_nav_cny": nav["stress"],
                "cash_base_cny": cash["base"],
                "cash_stress_cny": cash["stress"],
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "position_count": len(positions),
                "pending_exit_count": sum(
                    position.pending_exit for position in positions.values()
                ),
                "quality_complete": True,
                "capacity_pass": True,
            }
        )
        prior_nav = nav

    daily = pd.DataFrame(daily_rows)
    trade_columns = [
        "trade_date",
        "side",
        "bond_code",
        "units",
        "gross_notional_cny",
        "base_cost_rate",
        "stress_cost_rate",
        "base_cash_flow_cny",
        "stress_cash_flow_cny",
        "signal_date",
        "exit_signal_date",
        "entry_date",
        "delayed_exit_days",
        "capacity_fraction_at_entry",
        "open_gap",
        "double_low_score",
    ]
    trades = pd.DataFrame(trade_rows, columns=trade_columns)
    audit = {
        "initial_capital_cny": initial_capital,
        "final_base_nav_cny": float(daily["base_nav_cny"].iloc[-1]),
        "final_stress_nav_cny": float(daily["stress_nav_cny"].iloc[-1]),
        "rebalance_count": int(rebalance_count),
        "empty_target_rebalance_count": int(empty_target_rebalance_count),
        "retained_position_observation_count": int(retained_position_count),
        "pending_exit_reselected_count": int(pending_exit_reselected_count),
        "entry_candidate_count": int(entry_candidate_count),
        "entry_transaction_count": int(entry_transaction_count),
        "exit_transaction_count": int(exit_transaction_count),
        "invalid_entry_bar_skip_count": int(invalid_entry_bar_skip_count),
        "upper_gap_entry_skip_count": int(upper_gap_entry_skip_count),
        "minimum_notional_skip_count": int(minimum_notional_skip_count),
        "capacity_limited_entry_count": int(capacity_limited_entry_count),
        "maximum_position_limit_skip_count": int(maximum_position_limit_skip_count),
        "blocked_exit_attempt_count": int(blocked_exit_attempt_count),
        "stale_mark_count": int(stale_mark_count),
        "maximum_capacity_fraction": float(maximum_capacity_fraction),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_net_exposure),
        "maximum_position_count": int(maximum_position_count),
        "terminal_open_position_count": 0,
        "same_close_execution_used": False,
        "historical_order_queue_observed": False,
        "execution_cost_status": "MODELED_BASE_AND_STRESS_ALL_IN_CB_COSTS",
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
    """以UTF-8原子写入文本。"""

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
    """生成决策导向的可见期报告。"""

    evaluation = report.get("evaluation", {})
    metrics = evaluation.get("metrics", {})
    portfolio = report.get("data_audit", {}).get("portfolio", {})
    lines = [
        "# 中国可转债传统双低周频轮动V1可见期结果",
        "",
        f"状态：`{report['status']}`",
        "",
        "本结果只是可见历史公式族筛选，不能验证未来每年40个百分点净超额，也不授权Paper、Shadow、订单或实盘。",
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
            "封存复验未读取；Paper、Shadow、订单、券商连接、转股和实盘均关闭。",
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
    """仅运行一次冻结可见期，并写出不可覆盖的研究产物。"""

    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, benchmark, input_audit = load_visible_inputs(
        contract, signal_only=False
    )
    selections, signal_audit = build_weekly_selections(
        panel,
        master,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        selections,
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
        "FROZEN_VISIBLE_CB_FORMULA_DISCOVERY_WITH_CONSERVATIVE_ZERO_VARIANCE"
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    status = (
        "VISIBLE_PASS_SEALED_REPLICATION_AUTHORIZED_NOT_OPENED"
        if passed
        else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
    )
    report = {
        "schema_version": "1.0.0",
        "report_id": "CB_DOUBLE_LOW_ROTATION_V1_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "performance_metrics_available": True,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": {
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
        },
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
            "sealed_replication_authorized": passed,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "next_step": (
                "可见期硬门全部通过；本次仍不读取封存面板，下一步才可单独开启非外部盲样本的公式族复验"
                if passed
                else "冻结拒绝本候选；不打开2023—2026封存复验，不改变参数救回，继续登记新的独立机制"
            ),
        },
        "outputs": contract["outputs"],
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_selections"], selections)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
