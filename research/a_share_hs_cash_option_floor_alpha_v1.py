from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROTOCOL_ID = "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1"
EXPECTED_EVENT_COUNT = 21
QUALIFIED = "QUALIFIED_HIGH_FLOOR_SPREAD"
NOT_QUALIFIED = "NOT_QUALIFIED_HIGH_FLOOR_SPREAD"
NO_ENTRY = "NO_REPLICABLE_ENTRY_WINDOW"
NO_VIEW_MARKET = "NO_VIEW_MARKET_DATA"
ENTRY_FOUND = "ENTRY_FOUND"


class CashOptionFloorProtocolError(ValueError):
    """现金选择权底价协议或输入不满足冻结约束。"""


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise CashOptionFloorProtocolError(
            f"协议字段漂移：{field}，预期={expected!r}，实际={actual!r}"
        )


def _require_false(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not False:
        raise CashOptionFloorProtocolError(f"治理字段必须为 false：{key}")


def validate_protocol_config(config: Mapping[str, Any]) -> None:
    """验证所有影响结论的冻结参数，防止价格读取后被静默修改。"""

    _require_equal(config.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    _require_equal(config.get("evidence_cutoff"), "2026-08-14", "evidence_cutoff")
    _require_equal(config.get("research_scope"), "DISCOVERY_ONLY", "research_scope")

    upstream = config.get("upstream") or {}
    _require_equal(upstream.get("expected_event_count"), 21, "upstream.expected_event_count")
    _require_equal(
        upstream.get("required_terms_status"),
        "COMPLETE_A_SHARE_CASH_OPTION_TERMS",
        "upstream.required_terms_status",
    )

    universe = config.get("event_universe") or {}
    _require_equal(universe.get("all_events_must_receive_price_disposition"), True, "event_universe.all_events_must_receive_price_disposition")
    for key in ("liquidity_filter", "market_cap_filter", "adv_filter", "turnover_filter", "st_exclusion"):
        _require_false(universe, key)

    entry = config.get("clock_and_entry") or {}
    exact_entry = {
        "entry_search_start": "FIRST_EXCHANGE_OPEN_DATE_STRICTLY_AFTER_INFORMATION_DATE",
        "entry_search_end": "RIGHTS_REGISTRATION_DATE_INCLUSIVE",
        "entry_date": "FIRST_VALID_FILLABLE_SUBJECT_TRADING_DATE_IN_SEARCH_WINDOW",
        "entry_price": "RAW_UNADJUSTED_DAILY_HIGH",
        "board_lot_shares": 100,
        "positive_volume_required": True,
        "valid_ohlc_required": True,
        "pre_close_required": True,
        "one_price_up_day_fill_assumed": False,
        "one_price_up_days_are_skipped_until_registration_date": True,
        "later_refill_after_first_valid_fillable_day_allowed": False,
        "fractional_or_odd_lot_assumption_allowed": False,
    }
    for key, expected in exact_entry.items():
        _require_equal(entry.get(key), expected, f"clock_and_entry.{key}")

    market = config.get("market_data_contract") or {}
    exact_market = {
        "subject_primary_source": "TUSHARE_PRO_DAILY_RAW_UNADJUSTED",
        "subject_primary_endpoint": "daily",
        "independent_second_source_required_for_entry_row": True,
        "same_vendor_repackaging_counts_as_independent": False,
        "entry_ohlc_cross_source_max_absolute_difference_cny": 0.005,
        "entry_date_exact_match_required": True,
        "raw_response_preservation_required": True,
        "request_receipt_required": True,
        "sha256_required": True,
        "query_window_must_cover_every_exchange_open_date": True,
        "price_adjustment": "NONE_RAW_UNADJUSTED",
        "no_silent_source_substitution": True,
    }
    for key, expected in exact_market.items():
        _require_equal(market.get(key), expected, f"market_data_contract.{key}")

    screening = config.get("screening") or {}
    exact_screening = {
        "minimum_net_conditional_floor_return": 0.08,
        "buy_cost_bps": 50,
        "cash_exercise_and_settlement_cost_bps": 100,
        "minimum_fee_per_leg_cny": 5.0,
        "annual_simple_funding_rate": 0.08,
        "funding_day_count_basis": 365,
        "screen_settlement_buffer_calendar_days_after_application_end": 10,
        "screen_uses_future_success_information": False,
        "screen_uses_future_market_return": False,
        "screen_uses_liquidity": False,
    }
    for key, expected in exact_screening.items():
        _require_equal(screening.get(key), expected, f"screening.{key}")

    lifecycle = config.get("lifecycle_outcome") or {}
    exact_lifecycle = {
        "all_price_qualified_events_require_complete_official_lifecycle": True,
        "later_success_required_for_price_qualification": False,
        "cash_distribution_tax_haircut_fraction": 0.20,
        "success_cost_bps": 100,
        "failure_or_withdrawal_sale_cost_bps": 100,
        "failure_one_price_down_sell_assumed": False,
        "failure_exit_search_continues_until_first_fillable_sale": True,
        "actual_funding_rate": 0.08,
        "outlier_deletion_allowed": False,
        "winsorization_allowed": False,
    }
    for key, expected in exact_lifecycle.items():
        _require_equal(lifecycle.get(key), expected, f"lifecycle_outcome.{key}")

    benchmark = config.get("benchmark") or {}
    exact_benchmark = {
        "primary": "CSI300_PRICE_INDEX_000300_SH_RAW_UNADJUSTED",
        "primary_source": "TUSHARE_PRO_INDEX_DAILY",
        "symbol": "000300.SH",
        "entry_value": "RAW_LOW_ON_STRATEGY_ENTRY_DATE",
        "terminal_value": "RAW_HIGH_ON_STRATEGY_TERMINAL_DATE",
        "benchmark_costs": 0.0,
        "benchmark_dividends_included": False,
        "exact_endpoint_dates_required": True,
        "secondary_may_rescue_primary": False,
    }
    for key, expected in exact_benchmark.items():
        _require_equal(benchmark.get(key), expected, f"benchmark.{key}")

    bootstrap = config.get("bootstrap") or {}
    exact_bootstrap = {
        "unit": "QUALIFIED_EVENT",
        "repetitions": 10000,
        "seed": 20260824,
        "confidence_level": 0.95,
        "lower_quantile": 0.025,
    }
    for key, expected in exact_bootstrap.items():
        _require_equal(bootstrap.get(key), expected, f"bootstrap.{key}")

    gates = config.get("acceptance_gates") or {}
    exact_gates = {
        "minimum_price_qualified_events": 5,
        "minimum_distinct_announcement_years": 3,
        "minimum_distinct_eras": 2,
        "all_qualified_lifecycles_complete": True,
        "all_qualified_benchmarks_complete": True,
        "minimum_transaction_completion_rate": 0.80,
        "transaction_completion_wilson_95_lower_strictly_above": 0.50,
        "minimum_excess_win_rate": 0.80,
        "excess_win_wilson_95_lower_strictly_above": 0.50,
        "minimum_mean_net_realized_return": 0.08,
        "minimum_median_net_realized_return": 0.08,
        "minimum_mean_best_case_csi300_excess_return": 0.05,
        "minimum_median_best_case_csi300_excess_return": 0.05,
        "bootstrap_95_lower_mean_excess_strictly_above": 0.0,
        "minimum_excess_payoff_ratio": 2.0,
        "minimum_excess_profit_factor": 4.0,
        "minimum_worst_single_excess_return": -0.20,
        "both_chronological_half_mean_excess_strictly_above": 0.0,
        "leave_best_event_out_mean_excess_strictly_above": 0.0,
        "minimum_positive_announcement_year_fraction": 0.75,
        "maximum_single_winner_share_of_positive_excess": 0.50,
    }
    for key, expected in exact_gates.items():
        _require_equal(gates.get(key), expected, f"acceptance_gates.{key}")

    governance = config.get("governance") or {}
    _require_equal(
        governance.get("market_schema_and_coverage_metadata_inspected_before_freeze"),
        True,
        "governance.market_schema_and_coverage_metadata_inspected_before_freeze",
    )
    for key in (
        "cash_option_eligible_event_set_may_change",
        "cash_option_terms_may_change",
        "entry_rule_may_change_after_freeze",
        "costs_may_change_after_freeze",
        "floor_threshold_may_change_after_freeze",
        "benchmark_may_change_after_freeze",
        "acceptance_gates_may_change_after_freeze",
        "parameter_search_allowed",
        "market_price_value_read_before_freeze",
        "future_subject_return_read_before_freeze",
        "price_snippets_incidentally_visible_in_official_term_documents_used_for_selection",
        "liquidity_filter_applied",
        "market_cap_filter_applied",
        "observed_spread_filter_applied_before_freeze",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
    ):
        _require_false(governance, key)


def as_date(value: Any, field: str) -> date:
    try:
        parsed = pd.Timestamp(value)
    except Exception as error:  # noqa: BLE001 - 保留字段上下文
        raise CashOptionFloorProtocolError(f"{field} 不是有效日期：{value!r}") from error
    if pd.isna(parsed):
        raise CashOptionFloorProtocolError(f"{field} 不得为空")
    return parsed.date()


def positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CashOptionFloorProtocolError(f"{field} 不是数值：{value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise CashOptionFloorProtocolError(f"{field} 必须是有限正数：{value!r}")
    return number


def nonnegative_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CashOptionFloorProtocolError(f"{field} 不是数值：{value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise CashOptionFloorProtocolError(f"{field} 必须是有限非负数：{value!r}")
    return number


def inclusive_calendar_days(start: Any, end: Any) -> int:
    start_date = as_date(start, "start")
    end_date = as_date(end, "end")
    if end_date < start_date:
        raise CashOptionFloorProtocolError("终值日早于入场日")
    return (end_date - start_date).days + 1


def leg_fee(notional: float, bps: float, minimum_fee: float) -> float:
    notional_value = positive_number(notional, "notional")
    rate_bps = nonnegative_number(bps, "bps")
    minimum = nonnegative_number(minimum_fee, "minimum_fee")
    return max(minimum, notional_value * rate_bps / 10_000.0)


def entry_outlay(entry_high: float, config: Mapping[str, Any]) -> dict[str, float]:
    high = positive_number(entry_high, "entry_high")
    lot = int(config["clock_and_entry"]["board_lot_shares"])
    screening = config["screening"]
    notional = lot * high
    fee = leg_fee(
        notional,
        float(screening["buy_cost_bps"]),
        float(screening["minimum_fee_per_leg_cny"]),
    )
    return {"entry_notional": notional, "buy_fee": fee, "entry_outlay": notional + fee}


def financed_outlay(
    entry_high: float,
    entry_date: Any,
    terminal_date: Any,
    annual_rate: float,
    config: Mapping[str, Any],
) -> dict[str, float | int]:
    initial = entry_outlay(entry_high, config)
    days = inclusive_calendar_days(entry_date, terminal_date)
    basis = int(config["screening"]["funding_day_count_basis"])
    rate = nonnegative_number(annual_rate, "annual_rate")
    factor = 1.0 + rate * days / basis
    return {
        **initial,
        "funding_days": days,
        "funding_factor": factor,
        "funded_entry_outlay": initial["entry_outlay"] * factor,
    }


def valid_ohlc(row: Mapping[str, Any], tolerance: float = 1e-8) -> bool:
    try:
        open_price = positive_number(row["raw_open"], "raw_open")
        high = positive_number(row["raw_high"], "raw_high")
        low = positive_number(row["raw_low"], "raw_low")
        close = positive_number(row["raw_close"], "raw_close")
        pre_close = positive_number(row["pre_close"], "pre_close")
    except (KeyError, CashOptionFloorProtocolError):
        return False
    return (
        high + tolerance >= max(open_price, low, close)
        and low - tolerance <= min(open_price, high, close)
        and pre_close > 0
    )


def one_price_up_day(row: Mapping[str, Any], tolerance: float = 1e-8) -> bool:
    if not valid_ohlc(row, tolerance=tolerance):
        return False
    prices = [float(row[key]) for key in ("raw_open", "raw_high", "raw_low", "raw_close")]
    return max(prices) - min(prices) <= tolerance and prices[-1] > float(row["pre_close"]) + tolerance


def select_entry_row(
    event: Mapping[str, Any],
    subject_daily: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """按冻结时钟选择首个可成交日；不读取或比较窗口后的价格。"""

    required_columns = {
        "ts_code",
        "date",
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "volume",
    }
    missing = sorted(required_columns.difference(subject_daily.columns))
    if missing:
        raise CashOptionFloorProtocolError(f"标的日线缺字段：{missing}")
    ts_code = str(event["ts_code"])
    announcement_date = as_date(event["effective_announcement_date"], "effective_announcement_date")
    registration_date = as_date(event["rights_registration_date"], "rights_registration_date")
    if registration_date <= announcement_date:
        return {
            "entry_status": NO_ENTRY,
            "reason_code": "NO_CALENDAR_DAY_STRICTLY_AFTER_ANNOUNCEMENT_BEFORE_OR_ON_REGISTRATION",
            "one_price_up_days_skipped": 0,
        }

    frame = subject_daily.loc[subject_daily["ts_code"].astype(str).eq(ts_code)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    frame = frame.loc[
        frame["date"].gt(announcement_date) & frame["date"].le(registration_date)
    ].sort_values("date", kind="stable")
    if frame["date"].duplicated().any():
        raise CashOptionFloorProtocolError(f"入场窗口存在重复标的日期：{ts_code}")

    skipped_one_price_up = 0
    observed_rows = 0
    for row in frame.to_dict(orient="records"):
        observed_rows += 1
        try:
            volume = float(row["volume"])
        except (TypeError, ValueError):
            return {
                "entry_status": NO_VIEW_MARKET,
                "reason_code": "NON_NUMERIC_VOLUME_IN_ENTRY_WINDOW",
                "one_price_up_days_skipped": skipped_one_price_up,
            }
        if not math.isfinite(volume):
            return {
                "entry_status": NO_VIEW_MARKET,
                "reason_code": "NON_FINITE_VOLUME_IN_ENTRY_WINDOW",
                "one_price_up_days_skipped": skipped_one_price_up,
            }
        if volume <= 0:
            continue
        if not valid_ohlc(row, tolerance=float(config["clock_and_entry"]["one_price_numeric_tolerance"])):
            return {
                "entry_status": NO_VIEW_MARKET,
                "reason_code": "INVALID_POSITIVE_VOLUME_OHLC_OR_PRE_CLOSE",
                "one_price_up_days_skipped": skipped_one_price_up,
            }
        if one_price_up_day(row, tolerance=float(config["clock_and_entry"]["one_price_numeric_tolerance"])):
            skipped_one_price_up += 1
            continue
        return {
            "entry_status": ENTRY_FOUND,
            "reason_code": "FIRST_VALID_FILLABLE_TRADING_DAY",
            "entry_date": row["date"].isoformat(),
            "entry_high": float(row["raw_high"]),
            "entry_source": row.get("price_source"),
            "one_price_up_days_skipped": skipped_one_price_up,
            "observed_subject_rows_through_entry": observed_rows,
        }
    return {
        "entry_status": NO_ENTRY,
        "reason_code": "NO_VALID_FILLABLE_SUBJECT_ROW_THROUGH_REGISTRATION_DATE",
        "one_price_up_days_skipped": skipped_one_price_up,
        "observed_subject_rows_in_window": observed_rows,
    }


def screen_floor_metrics(
    entry_high: float,
    cash_option_price: float,
    entry_date: Any,
    application_end_date: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    screening = config["screening"]
    cash_price = positive_number(cash_option_price, "cash_option_price")
    application_end = as_date(application_end_date, "application_end_date")
    buffer_days = int(screening["screen_settlement_buffer_calendar_days_after_application_end"])
    assumed_terminal = application_end + timedelta(days=buffer_days)
    funded = financed_outlay(
        entry_high,
        entry_date,
        assumed_terminal,
        float(screening["annual_simple_funding_rate"]),
        config,
    )
    lot = int(config["clock_and_entry"]["board_lot_shares"])
    cash_notional = lot * cash_price
    exercise_fee = leg_fee(
        cash_notional,
        float(screening["cash_exercise_and_settlement_cost_bps"]),
        float(screening["minimum_fee_per_leg_cny"]),
    )
    net_cash = cash_notional - exercise_fee
    net_floor_return = net_cash / float(funded["funded_entry_outlay"]) - 1.0
    threshold = float(screening["minimum_net_conditional_floor_return"])
    return {
        **funded,
        "announced_cash_price": cash_price,
        "cash_notional": cash_notional,
        "cash_exercise_and_settlement_fee": exercise_fee,
        "screen_net_cash": net_cash,
        "assumed_screen_terminal_date": assumed_terminal.isoformat(),
        "net_conditional_floor_return": net_floor_return,
        "price_disposition": QUALIFIED if net_floor_return >= threshold else NOT_QUALIFIED,
    }


def screen_event(
    event: Mapping[str, Any],
    subject_daily: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    entry = select_entry_row(event, subject_daily, config)
    base = {
        "event_id": str(event["event_id"]),
        "ts_code": str(event["ts_code"]),
        "effective_announcement_date": as_date(event["effective_announcement_date"], "effective_announcement_date").isoformat(),
        "rights_registration_date": as_date(event["rights_registration_date"], "rights_registration_date").isoformat(),
        "application_end_date": as_date(event["application_end_date"], "application_end_date").isoformat(),
        **entry,
    }
    if entry["entry_status"] == NO_VIEW_MARKET:
        return {**base, "price_disposition": NO_VIEW_MARKET}
    if entry["entry_status"] == NO_ENTRY:
        return {**base, "price_disposition": NO_ENTRY}
    metrics = screen_floor_metrics(
        entry["entry_high"],
        event["cash_option_price"],
        entry["entry_date"],
        event["application_end_date"],
        config,
    )
    return {**base, **metrics}


def successful_cash_outcome_metrics(
    *,
    entry_high: float,
    entry_date: Any,
    final_cash_option_price: float,
    cash_availability_date: Any,
    adjusted_share_multiplier: float = 1.0,
    entitled_cash_distribution_per_original_share: float = 0.0,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle = config["lifecycle_outcome"]
    lot = int(config["clock_and_entry"]["board_lot_shares"])
    multiplier = positive_number(adjusted_share_multiplier, "adjusted_share_multiplier")
    final_price = positive_number(final_cash_option_price, "final_cash_option_price")
    distribution = nonnegative_number(
        entitled_cash_distribution_per_original_share,
        "entitled_cash_distribution_per_original_share",
    )
    option_notional = lot * multiplier * final_price
    fee = leg_fee(
        option_notional,
        float(lifecycle["success_cost_bps"]),
        float(config["screening"]["minimum_fee_per_leg_cny"]),
    )
    net_distribution = lot * distribution * (
        1.0 - float(lifecycle["cash_distribution_tax_haircut_fraction"])
    )
    terminal_cash = option_notional - fee + net_distribution
    funded = financed_outlay(
        entry_high,
        entry_date,
        cash_availability_date,
        float(lifecycle["actual_funding_rate"]),
        config,
    )
    realized_return = terminal_cash / float(funded["funded_entry_outlay"]) - 1.0
    return {
        **funded,
        "outcome_type": "SUCCESS_CASH_EXERCISED_AND_SETTLED",
        "adjusted_share_multiplier": multiplier,
        "final_cash_option_price": final_price,
        "option_notional": option_notional,
        "terminal_leg_fee": fee,
        "net_cash_distribution": net_distribution,
        "terminal_cash": terminal_cash,
        "realized_net_return": realized_return,
    }


def failed_or_withdrawn_outcome_metrics(
    *,
    entry_high: float,
    entry_date: Any,
    exit_open: float,
    exit_date: Any,
    adjusted_share_multiplier: float = 1.0,
    entitled_cash_distribution_per_original_share: float = 0.0,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle = config["lifecycle_outcome"]
    lot = int(config["clock_and_entry"]["board_lot_shares"])
    multiplier = positive_number(adjusted_share_multiplier, "adjusted_share_multiplier")
    open_price = positive_number(exit_open, "exit_open")
    distribution = nonnegative_number(
        entitled_cash_distribution_per_original_share,
        "entitled_cash_distribution_per_original_share",
    )
    sale_notional = lot * multiplier * open_price
    fee = leg_fee(
        sale_notional,
        float(lifecycle["failure_or_withdrawal_sale_cost_bps"]),
        float(config["screening"]["minimum_fee_per_leg_cny"]),
    )
    net_distribution = lot * distribution * (
        1.0 - float(lifecycle["cash_distribution_tax_haircut_fraction"])
    )
    terminal_cash = sale_notional - fee + net_distribution
    funded = financed_outlay(
        entry_high,
        entry_date,
        exit_date,
        float(lifecycle["actual_funding_rate"]),
        config,
    )
    realized_return = terminal_cash / float(funded["funded_entry_outlay"]) - 1.0
    return {
        **funded,
        "outcome_type": "FAILED_WITHDRAWN_OR_TERMINATED_MARKET_EXIT",
        "adjusted_share_multiplier": multiplier,
        "exit_open": open_price,
        "sale_notional": sale_notional,
        "terminal_leg_fee": fee,
        "net_cash_distribution": net_distribution,
        "terminal_cash": terminal_cash,
        "realized_net_return": realized_return,
    }


def best_case_csi300_return(entry_index_low: float, terminal_index_high: float) -> float:
    low = positive_number(entry_index_low, "entry_index_low")
    high = positive_number(terminal_index_high, "terminal_index_high")
    return high / low - 1.0


def wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise CashOptionFloorProtocolError("Wilson 区间计数非法")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - radius) / denominator


def bootstrap_mean_lower(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    lower_quantile: float,
) -> float:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise CashOptionFloorProtocolError("自助法输入必须是一维有限非空数组")
    if repetitions <= 0 or not 0.0 < lower_quantile < 0.5:
        raise CashOptionFloorProtocolError("自助法参数非法")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, lower_quantile))


def assign_era(value: Any, config: Mapping[str, Any]) -> str:
    event_date = as_date(value, "effective_announcement_date")
    matches = [
        str(item["era"])
        for item in config["era_definitions"]
        if as_date(item["start"], "era.start") <= event_date <= as_date(item["end"], "era.end")
    ]
    if len(matches) != 1:
        raise CashOptionFloorProtocolError(
            f"公告日期必须且只能落入一个冻结年代：{event_date.isoformat()}，匹配={matches}"
        )
    return matches[0]


def _ratio_or_infinity(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    if numerator > 0:
        return math.inf
    return 0.0


def _json_metric(value: float) -> float | str:
    if math.isinf(value):
        return "INF"
    return float(value)


def evaluate_study(
    price_screen: pd.DataFrame,
    lifecycle_outcomes: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """按冻结停止规则评价完整事件集；NO_VIEW 优先于局部可见样本。"""

    validate_protocol_config(config)
    required_screen = {"event_id", "price_disposition", "effective_announcement_date"}
    missing_screen = sorted(required_screen.difference(price_screen.columns))
    if missing_screen:
        raise CashOptionFloorProtocolError(f"价格筛选缺字段：{missing_screen}")
    if len(price_screen) != EXPECTED_EVENT_COUNT:
        raise CashOptionFloorProtocolError(
            f"价格筛选必须恰好有 {EXPECTED_EVENT_COUNT} 行，实际={len(price_screen)}"
        )
    event_ids = price_screen["event_id"].astype(str)
    if event_ids.duplicated().any():
        raise CashOptionFloorProtocolError("价格筛选事件ID重复")
    allowed = set(config["event_universe"]["allowed_price_dispositions"])
    actual_dispositions = set(price_screen["price_disposition"].astype(str))
    if not actual_dispositions.issubset(allowed):
        raise CashOptionFloorProtocolError(
            f"价格处置包含非法状态：{sorted(actual_dispositions - allowed)}"
        )

    status_contract = config["status_contract"]
    disposition_counts = {
        str(key): int(value)
        for key, value in price_screen["price_disposition"].value_counts().to_dict().items()
    }
    market_no_view_count = int(price_screen["price_disposition"].eq(NO_VIEW_MARKET).sum())
    if market_no_view_count:
        return {
            "status": status_contract["incomplete_price_universe"],
            "price_event_count": len(price_screen),
            "price_disposition_counts": disposition_counts,
            "qualified_event_count": int(price_screen["price_disposition"].eq(QUALIFIED).sum()),
            "market_no_view_count": market_no_view_count,
            "gates_evaluated": False,
        }

    qualified = price_screen.loc[price_screen["price_disposition"].eq(QUALIFIED)].copy()
    qualified_count = len(qualified)
    if qualified_count == 0:
        return {
            "status": status_contract["zero_qualified"],
            "price_event_count": len(price_screen),
            "price_disposition_counts": disposition_counts,
            "qualified_event_count": 0,
            "gates_evaluated": False,
        }
    minimum_qualified = int(config["acceptance_gates"]["minimum_price_qualified_events"])
    if qualified_count < minimum_qualified:
        return {
            "status": status_contract["insufficient_qualified"],
            "price_event_count": len(price_screen),
            "price_disposition_counts": disposition_counts,
            "qualified_event_count": qualified_count,
            "minimum_required": minimum_qualified,
            "gates_evaluated": False,
        }

    required_outcome = {
        "event_id",
        "effective_announcement_date",
        "terminal_date",
        "lifecycle_complete",
        "benchmark_complete",
        "transaction_completed",
        "realized_net_return",
        "benchmark_best_case_return",
    }
    missing_outcome = sorted(required_outcome.difference(lifecycle_outcomes.columns))
    if missing_outcome:
        raise CashOptionFloorProtocolError(f"生命周期结果缺字段：{missing_outcome}")
    outcomes = lifecycle_outcomes.copy()
    outcomes["event_id"] = outcomes["event_id"].astype(str)
    if outcomes["event_id"].duplicated().any():
        raise CashOptionFloorProtocolError("生命周期结果事件ID重复")
    qualified_ids = set(qualified["event_id"].astype(str))
    outcome_ids = set(outcomes["event_id"])
    if not qualified_ids.issubset(outcome_ids):
        return {
            "status": status_contract["incomplete_qualified_lifecycle"],
            "qualified_event_count": qualified_count,
            "missing_event_ids": sorted(qualified_ids - outcome_ids),
            "gates_evaluated": False,
        }
    outcomes = outcomes.loc[outcomes["event_id"].isin(qualified_ids)].copy()
    if len(outcomes) != qualified_count or not outcomes["lifecycle_complete"].fillna(False).astype(bool).all():
        incomplete = outcomes.loc[
            ~outcomes["lifecycle_complete"].fillna(False).astype(bool), "event_id"
        ].astype(str).tolist()
        return {
            "status": status_contract["incomplete_qualified_lifecycle"],
            "qualified_event_count": qualified_count,
            "incomplete_event_ids": sorted(incomplete),
            "gates_evaluated": False,
        }
    if not outcomes["benchmark_complete"].fillna(False).astype(bool).all():
        incomplete = outcomes.loc[
            ~outcomes["benchmark_complete"].fillna(False).astype(bool), "event_id"
        ].astype(str).tolist()
        return {
            "status": status_contract["incomplete_qualified_benchmark"],
            "qualified_event_count": qualified_count,
            "incomplete_event_ids": sorted(incomplete),
            "gates_evaluated": False,
        }

    for column in ("realized_net_return", "benchmark_best_case_return"):
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
        if not np.isfinite(outcomes[column].to_numpy(dtype=float)).all():
            raise CashOptionFloorProtocolError(f"生命周期结果存在非有限数：{column}")
    outcomes["effective_announcement_date"] = pd.to_datetime(
        outcomes["effective_announcement_date"], errors="raise"
    ).dt.date
    outcomes["terminal_date"] = pd.to_datetime(outcomes["terminal_date"], errors="raise").dt.date
    outcomes["announcement_year"] = outcomes["effective_announcement_date"].map(lambda value: value.year)
    outcomes["era"] = outcomes["effective_announcement_date"].map(lambda value: assign_era(value, config))
    distinct_years = int(outcomes["announcement_year"].nunique())
    distinct_eras = int(outcomes["era"].nunique())
    gates_config = config["acceptance_gates"]
    if (
        distinct_years < int(gates_config["minimum_distinct_announcement_years"])
        or distinct_eras < int(gates_config["minimum_distinct_eras"])
    ):
        return {
            "status": status_contract["insufficient_year_or_era_span"],
            "qualified_event_count": qualified_count,
            "distinct_announcement_years": distinct_years,
            "distinct_eras": distinct_eras,
            "gates_evaluated": False,
        }

    outcomes["excess_return"] = (
        outcomes["realized_net_return"] - outcomes["benchmark_best_case_return"]
    )
    outcomes = outcomes.sort_values(
        ["effective_announcement_date", "event_id"], kind="stable"
    ).reset_index(drop=True)
    realized = outcomes["realized_net_return"].to_numpy(dtype=float)
    excess = outcomes["excess_return"].to_numpy(dtype=float)
    transaction_completed = outcomes["transaction_completed"].fillna(False).astype(bool).to_numpy()
    excess_wins = excess > 0.0
    completion_count = int(transaction_completed.sum())
    win_count = int(excess_wins.sum())
    total = len(outcomes)
    completion_rate = completion_count / total
    win_rate = win_count / total
    completion_wilson = wilson_lower_bound(completion_count, total)
    win_wilson = wilson_lower_bound(win_count, total)
    positive = excess[excess > 0.0]
    negative_abs = np.abs(excess[excess < 0.0])
    payoff_ratio = _ratio_or_infinity(
        float(positive.mean()) if len(positive) else 0.0,
        float(negative_abs.mean()) if len(negative_abs) else 0.0,
    )
    profit_factor = _ratio_or_infinity(float(positive.sum()), float(negative_abs.sum()))
    bootstrap = config["bootstrap"]
    bootstrap_lower = bootstrap_mean_lower(
        excess,
        int(bootstrap["repetitions"]),
        int(bootstrap["seed"]),
        float(bootstrap["lower_quantile"]),
    )
    split = total // 2
    first_half_mean = float(excess[:split].mean())
    second_half_mean = float(excess[split:].mean())
    best_index = int(np.argmax(excess))
    leave_best_out_mean = float(np.delete(excess, best_index).mean())
    yearly_means = outcomes.groupby("announcement_year", sort=True)["excess_return"].mean()
    positive_year_fraction = float(yearly_means.gt(0.0).mean())
    positive_sum = float(positive.sum())
    maximum_winner_share = float(positive.max() / positive_sum) if positive_sum > 0 else 1.0

    metrics: dict[str, Any] = {
        "qualified_event_count": qualified_count,
        "distinct_announcement_years": distinct_years,
        "distinct_eras": distinct_eras,
        "transaction_completion_count": completion_count,
        "transaction_completion_rate": completion_rate,
        "transaction_completion_wilson_95_lower": completion_wilson,
        "excess_win_count": win_count,
        "excess_win_rate": win_rate,
        "excess_win_wilson_95_lower": win_wilson,
        "mean_net_realized_return": float(realized.mean()),
        "median_net_realized_return": float(np.median(realized)),
        "mean_best_case_csi300_excess_return": float(excess.mean()),
        "median_best_case_csi300_excess_return": float(np.median(excess)),
        "bootstrap_95_lower_mean_excess": bootstrap_lower,
        "excess_payoff_ratio": _json_metric(payoff_ratio),
        "excess_profit_factor": _json_metric(profit_factor),
        "worst_single_excess_return": float(excess.min()),
        "first_chronological_half_mean_excess": first_half_mean,
        "second_chronological_half_mean_excess": second_half_mean,
        "leave_best_event_out_mean_excess": leave_best_out_mean,
        "positive_announcement_year_fraction": positive_year_fraction,
        "maximum_single_winner_share_of_positive_excess": maximum_winner_share,
    }
    gates = {
        "transaction_completion_rate": completion_rate >= float(gates_config["minimum_transaction_completion_rate"]),
        "transaction_completion_wilson": completion_wilson > float(gates_config["transaction_completion_wilson_95_lower_strictly_above"]),
        "excess_win_rate": win_rate >= float(gates_config["minimum_excess_win_rate"]),
        "excess_win_wilson": win_wilson > float(gates_config["excess_win_wilson_95_lower_strictly_above"]),
        "mean_net_realized_return": float(realized.mean()) >= float(gates_config["minimum_mean_net_realized_return"]),
        "median_net_realized_return": float(np.median(realized)) >= float(gates_config["minimum_median_net_realized_return"]),
        "mean_excess_return": float(excess.mean()) >= float(gates_config["minimum_mean_best_case_csi300_excess_return"]),
        "median_excess_return": float(np.median(excess)) >= float(gates_config["minimum_median_best_case_csi300_excess_return"]),
        "bootstrap_lower_mean_excess": bootstrap_lower > float(gates_config["bootstrap_95_lower_mean_excess_strictly_above"]),
        "excess_payoff_ratio": payoff_ratio >= float(gates_config["minimum_excess_payoff_ratio"]),
        "excess_profit_factor": profit_factor >= float(gates_config["minimum_excess_profit_factor"]),
        "worst_single_excess_return": float(excess.min()) >= float(gates_config["minimum_worst_single_excess_return"]),
        "first_chronological_half_mean_excess": first_half_mean > float(gates_config["both_chronological_half_mean_excess_strictly_above"]),
        "second_chronological_half_mean_excess": second_half_mean > float(gates_config["both_chronological_half_mean_excess_strictly_above"]),
        "leave_best_event_out_mean_excess": leave_best_out_mean > float(gates_config["leave_best_event_out_mean_excess_strictly_above"]),
        "positive_announcement_year_fraction": positive_year_fraction >= float(gates_config["minimum_positive_announcement_year_fraction"]),
        "winner_concentration": maximum_winner_share <= float(gates_config["maximum_single_winner_share_of_positive_excess"]),
    }
    pass_all = all(gates.values())
    return {
        "status": status_contract["pass"] if pass_all else status_contract["fail"],
        "price_event_count": len(price_screen),
        "price_disposition_counts": disposition_counts,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": sorted(key for key, passed in gates.items() if not passed),
        "gates_evaluated": True,
        "shadow_authorized": False,
        "live_authorized": False,
    }
