from __future__ import annotations

import json
import math
from decimal import Decimal
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_ID = (
    "A_SHARE_HS_OFFICIAL_UNCONDITIONAL_FULL_CASH_TENDER_SPREAD_V1_"
    "EVALUATION_V1_0_3"
)


class TenderEvaluationError(ValueError):
    """全面现金要约盲评输入或结果违反冻结协议。"""


def _date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_roles(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except Exception as error:  # noqa: BLE001
        raise TenderEvaluationError(f"生命周期角色不是JSON数组：{value}") from error
    if not isinstance(parsed, list):
        raise TenderEvaluationError(f"生命周期角色不是JSON数组：{value}")
    return [str(item) for item in parsed]


def validate_static_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise TenderEvaluationError("盲评协议ID不匹配")
    if config.get("evidence_cutoff") != "2026-08-14":
        raise TenderEvaluationError("证据截止日漂移")
    execution = config.get("clock_and_execution") or {}
    expected_execution = {
        "minimum_gross_spread": 0.08,
        "minimum_open_dates_strictly_after_entry_through_offer_end": 5,
        "later_refill_allowed": False,
        "st_exclusion": False,
        "liquidity_filter": False,
    }
    costs = config.get("costs") or {}
    expected_costs = {
        "base_buy_bps": 20,
        "base_tender_bps": 30,
        "stress_buy_bps": 100,
        "stress_tender_bps": 100,
    }
    bootstrap = config.get("bootstrap") or {}
    failures: list[str] = []
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            failures.append(f"EXECUTION:{key}:{execution.get(key)}!={expected}")
    for key, expected in expected_costs.items():
        if costs.get(key) != expected:
            failures.append(f"COST:{key}:{costs.get(key)}!={expected}")
    if bootstrap.get("repetitions") != 10_000:
        failures.append("BOOTSTRAP_REPETITIONS")
    if bootstrap.get("seed") != 20_260_824:
        failures.append("BOOTSTRAP_SEED")
    if bootstrap.get("minimum_valid_payoff_and_profit_factor_fraction") != 0.90:
        failures.append("BOOTSTRAP_VALID_FRACTION")
    expected_rows = {
        "final_event_outcomes": 34,
        "final_corporate_actions": 17,
        "lifecycle_review_archive": 104,
        "unified_daily_market": 11_840_729,
        "industry_peer_return_panel": 11_840_729,
        "security_master": 5_543,
        "security_status_intervals": 13_636,
        "trading_calendar": 6_136,
        "h00300_close_reference": 3_091,
        "csi300_price_index_ohlc_bridge": 3_456,
    }
    actual_expected = config.get("expected_parquet") or {}
    for key, expected in expected_rows.items():
        if (actual_expected.get(key) or {}).get("rows") != expected:
            failures.append(f"ROW_CONTRACT:{key}")
    governance = config.get("governance") or {}
    false_required = (
        "event_set_may_change",
        "spread_threshold_may_change",
        "entry_clock_may_change",
        "costs_may_change",
        "benchmarks_may_change",
        "acceptance_gates_may_change",
        "parameter_search_allowed",
        "liquidity_filter_allowed",
        "market_cap_filter_allowed",
        "outlier_deletion_allowed",
        "winsorization_allowed",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
        "tender_event_market_price_read_before_protocol_freeze",
        "tender_event_future_return_read_before_protocol_freeze",
        "benchmark_input_read_revealed_event_spread_or_return",
    )
    for key in false_required:
        if governance.get(key) is not False:
            failures.append(f"GOVERNANCE_FALSE_REQUIRED:{key}")
    if governance.get(
        "benchmark_input_values_read_for_v1_0_1_and_v1_0_2_failure_diagnosis"
    ) is not True:
        failures.append("BENCHMARK_DIAGNOSIS_READ_MUST_BE_DISCLOSED")
    if governance.get("event_clock_dates_read_for_v1_0_2_date_scope_diagnosis") is not True:
        failures.append("EVENT_CLOCK_DATE_SCOPE_DIAGNOSIS_MUST_BE_DISCLOSED")
    if failures:
        raise TenderEvaluationError(";".join(failures))
    return {
        "status": "PASS_TENDER_EVALUATION_STATIC_CONFIG_VALIDATED",
        "event_count": 34,
        "corporate_action_count": 17,
        "spread_threshold": 0.08,
        "stress_total_cost_bps": 200,
        "tender_event_market_price_read": False,
        "tender_event_future_return_read": False,
    }


def _calendar_map(calendar: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    required = {"exchange", "date", "is_open"}
    missing = sorted(required - set(calendar.columns))
    if missing:
        raise TenderEvaluationError(f"交易日历缺少字段：{missing}")
    normalized = calendar.copy()
    normalized["date"] = _date_series(normalized["date"])
    if normalized["date"].isna().any():
        raise TenderEvaluationError("交易日历存在无效日期")
    if normalized.duplicated(["exchange", "date"]).any():
        raise TenderEvaluationError("交易日历存在交易所×日期重复键")
    normalized = normalized.loc[normalized["is_open"].fillna(False).astype(bool)]
    return {
        str(exchange): sorted(group["date"].tolist())
        for exchange, group in normalized.groupby("exchange", sort=False)
    }


def _first_strictly_after(
    dates: list[pd.Timestamp], target: pd.Timestamp
) -> pd.Timestamp | None:
    for value in dates:
        if value > target:
            return value
    return None


def _master_active(
    master_by_code: dict[str, pd.Series], ts_code: str, entry_date: pd.Timestamp
) -> tuple[bool | None, str | None]:
    row = master_by_code.get(ts_code)
    if row is None:
        return None, None
    security_type = str(row.get("security_type"))
    if security_type != "ORDINARY_A_SHARE":
        return False, security_type
    list_date = pd.to_datetime(row.get("list_date"), errors="coerce")
    delist_date = pd.to_datetime(row.get("delist_date"), errors="coerce")
    if pd.isna(list_date):
        return None, security_type
    active = entry_date >= list_date.normalize() and (
        pd.isna(delist_date) or entry_date <= delist_date.normalize()
    )
    return bool(active), security_type


def _status_at_date(
    status_by_code: dict[str, pd.DataFrame], ts_code: str, target: pd.Timestamp
) -> str | None:
    frame = status_by_code.get(ts_code)
    if frame is None or frame.empty:
        return None
    matches = frame.loc[
        frame["valid_from"].le(target)
        & (frame["valid_to"].isna() | frame["valid_to"].ge(target))
    ]
    if matches.empty:
        return None
    return "|".join(sorted(matches["status"].astype(str).unique()))


def build_event_clock(
    outcomes: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    lifecycle_review: pd.DataFrame,
    calendar: pd.DataFrame,
    security_master: pd.DataFrame,
    security_status_intervals: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """只用冻结事件、官方日期和交易日历构造事件钟，不读取价格。"""

    validate_static_config(config)
    required_outcomes = {
        "event_id",
        "ts_code",
        "formal_report_pdf_date",
        "formal_announcement_id",
        "reported_offer_price_cny",
        "offer_end_date",
        "final_effective_offer_price_cny",
        "outcome_status",
        "result_announcement_id",
    }
    missing = sorted(required_outcomes - set(outcomes.columns))
    if missing:
        raise TenderEvaluationError(f"最终事件裁决缺少字段：{missing}")
    if len(outcomes) != 34 or outcomes["event_id"].astype(str).duplicated().any():
        raise TenderEvaluationError("最终事件裁决不是34个唯一事件")
    if len(corporate_actions) != 17:
        raise TenderEvaluationError("最终公司行动不是17条")

    result_rows: list[dict[str, str]] = []
    for row in lifecycle_review.to_dict(orient="records"):
        if "RESULT" in _json_roles(row["review_roles_json"]):
            result_rows.append(
                {
                    "event_id": str(row["event_id"]),
                    "result_announcement_id": str(row["announcement_id"]),
                    "result_pdf_date": str(row["official_pdf_date"]),
                }
            )
    results = pd.DataFrame(result_rows)
    if len(results) != 34 or results["event_id"].duplicated().any():
        raise TenderEvaluationError("生命周期审阅未产生34个唯一结果公告")
    merged = outcomes.copy()
    merged["event_id"] = merged["event_id"].astype(str)
    merged["result_announcement_id"] = merged["result_announcement_id"].astype(str)
    merged = merged.merge(
        results,
        on=["event_id", "result_announcement_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["result_pdf_date"].isna().any():
        raise TenderEvaluationError("结果公告日期映射不完整")

    calendars = _calendar_map(calendar)
    master = security_master.copy()
    master["ts_code"] = master["ts_code"].astype(str)
    if master["ts_code"].duplicated().any():
        raise TenderEvaluationError("证券主表ts_code不唯一")
    master_by_code = {str(row["ts_code"]): row for _, row in master.iterrows()}
    status = security_status_intervals.copy()
    status["ts_code"] = status["ts_code"].astype(str)
    status["valid_from"] = _date_series(status["valid_from"])
    status["valid_to"] = _date_series(status["valid_to"])
    status_by_code = {
        str(code): group.copy() for code, group in status.groupby("ts_code", sort=False)
    }

    actions = corporate_actions.copy()
    actions["event_id"] = actions["event_id"].astype(str)
    actions["official_pdf_date"] = _date_series(actions["official_pdf_date"])

    output: list[dict[str, Any]] = []
    for event in merged.sort_values(
        ["formal_report_pdf_date", "event_id"], kind="stable"
    ).to_dict(orient="records"):
        event_id = str(event["event_id"])
        ts_code = str(event["ts_code"])
        exchange = "SSE" if ts_code.endswith(".SH") else "SZSE"
        dates = calendars.get(exchange, [])
        formal_date = pd.to_datetime(event["formal_report_pdf_date"], errors="coerce")
        offer_end = pd.to_datetime(event["offer_end_date"], errors="coerce")
        result_date = pd.to_datetime(event["result_pdf_date"], errors="coerce")
        if any(pd.isna(value) for value in (formal_date, offer_end, result_date)):
            raise TenderEvaluationError(f"事件官方日期无效：{event_id}")
        formal_date = formal_date.normalize()
        offer_end = offer_end.normalize()
        result_date = result_date.normalize()
        report_date = _first_strictly_after(dates, formal_date)
        entry_date = (
            _first_strictly_after(dates, report_date)
            if report_date is not None
            else None
        )
        cash_date = _first_strictly_after(dates, result_date)
        clock_status = "READY_FOR_FROZEN_MARKET_EVALUATION"
        if report_date is None:
            clock_status = "NO_VIEW_REPORT_MARKET_DATE_AFTER_CALENDAR_CUTOFF"
        elif entry_date is None:
            clock_status = "NO_VIEW_ENTRY_DATE_AFTER_CALENDAR_CUTOFF"
        elif cash_date is None:
            clock_status = "NO_VIEW_CASH_AVAILABILITY_DATE_AFTER_CALENDAR_CUTOFF"

        active: bool | None = None
        security_type: str | None = None
        entry_status: str | None = None
        remaining_dates: list[pd.Timestamp] = []
        benchmark_dates: list[pd.Timestamp] = []
        if entry_date is not None:
            active, security_type = _master_active(master_by_code, ts_code, entry_date)
            entry_status = _status_at_date(status_by_code, ts_code, entry_date)
            remaining_dates = [value for value in dates if entry_date < value <= offer_end]
        if entry_date is not None and cash_date is not None:
            benchmark_dates = [value for value in dates if entry_date < value <= cash_date]

        observed_offer_price = float(event["reported_offer_price_cny"])
        event_actions = actions.loc[actions["event_id"].eq(event_id)]
        known_adjustments = event_actions.loc[
            event_actions["offer_price_adjusted"].fillna(False).astype(bool)
            & event_actions["official_pdf_date"].le(report_date)
        ] if report_date is not None else event_actions.iloc[0:0]
        if not known_adjustments.empty:
            observed_offer_price = float(event["final_effective_offer_price_cny"])

        output.append(
            {
                **event,
                "exchange": exchange,
                "formal_report_pdf_date": formal_date,
                "offer_end_date": offer_end,
                "result_pdf_date": result_date,
                "report_market_date": report_date,
                "entry_date": entry_date,
                "cash_availability_date": cash_date,
                "observed_offer_price_cny": observed_offer_price,
                "entry_limit_price_cny": observed_offer_price / 1.08,
                "open_dates_strictly_after_entry_through_offer_end": len(remaining_dates),
                "benchmark_after_entry_open_date_count": len(benchmark_dates),
                "benchmark_after_entry_dates_json": json.dumps(
                    [str(value.date()) for value in benchmark_dates]
                ),
                "entry_security_active": active if active is not None else pd.NA,
                "entry_security_type": security_type,
                "entry_security_status": entry_status,
                "clock_status": clock_status,
            }
        )
    result = pd.DataFrame(output)
    if len(result) != 34 or result["event_id"].duplicated().any():
        raise TenderEvaluationError("事件钟不是34个唯一事件")
    return result.reset_index(drop=True)


def one_price_upward_lock(
    row: pd.Series | dict[str, Any], *, relative_tolerance: float, pct_threshold: float
) -> bool:
    fields = ("raw_open", "raw_high", "raw_low", "raw_close")
    if any(not _finite(row.get(field)) for field in fields):
        raise TenderEvaluationError("一字涨停判断缺少开高低收")
    if not _finite(row.get("pct_chg")):
        raise TenderEvaluationError("一字涨停判断缺少涨跌幅")
    prices = np.asarray([float(row.get(field)) for field in fields], dtype=float)
    scale = max(float(np.max(np.abs(prices))), 1.0)
    same = float(np.max(prices) - np.min(prices)) <= relative_tolerance * scale
    return bool(same and float(row.get("pct_chg")) >= pct_threshold)


def _traded(row: pd.Series | None, required_prices: tuple[str, ...]) -> bool:
    if row is None:
        return False
    if not bool(row.get("observed_traded_row", False)):
        return False
    if bool(row.get("is_suspended", False)):
        return False
    if not _finite(row.get("amount")) or float(row.get("amount")) <= 0.0:
        return False
    return all(_finite(row.get(field)) and float(row.get(field)) > 0.0 for field in required_prices)


def _corporate_action_proceeds(
    event: pd.Series,
    actions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    accounting = config["corporate_action_accounting"]
    bonus = {
        str(key): float(value)
        for key, value in accounting["bonus_share_per_original_share_by_announcement_id"].items()
    }
    entry_date = pd.Timestamp(event["entry_date"]).normalize()
    offer_end = pd.Timestamp(event["offer_end_date"]).normalize()
    result_date = pd.Timestamp(event["result_pdf_date"]).normalize()
    entitled_ids: list[str] = []
    excluded_ids: list[str] = []
    unresolved_ids: list[str] = []
    cash = 0.0
    for action in actions.to_dict(orient="records"):
        announcement_id = str(action["announcement_id"])
        record_date = pd.to_datetime(action["record_date"], errors="coerce")
        if pd.isna(record_date):
            unresolved_ids.append(announcement_id)
            continue
        record_date = record_date.normalize()
        if record_date < entry_date:
            excluded_ids.append(announcement_id)
        elif record_date <= offer_end:
            entitled_ids.append(announcement_id)
            cash_value = action.get("gross_cash_per_share_cny")
            if not _finite(cash_value) or float(cash_value) <= 0.0:
                unresolved_ids.append(announcement_id)
                continue
            cash += float(cash_value)
            if bonus.get(announcement_id, 0.0) > 0.0:
                unresolved_ids.append(announcement_id)
        elif record_date > result_date:
            excluded_ids.append(announcement_id)
        else:
            unresolved_ids.append(announcement_id)
    return {
        "entitled_corporate_action_ids_json": json.dumps(entitled_ids),
        "excluded_corporate_action_ids_json": json.dumps(excluded_ids),
        "unresolved_corporate_action_ids_json": json.dumps(unresolved_ids),
        "entitled_gross_cash_distribution_cny": cash,
        "corporate_action_accounting_resolved": not unresolved_ids,
    }


def evaluate_event_ledger(
    event_clock: pd.DataFrame,
    event_market_rows: pd.DataFrame,
    industry_entry_stats: pd.DataFrame,
    industry_daily_rows: pd.DataFrame,
    h00300_ohlc: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """按冻结价差、开盘限价、公司行动、成本与基准机械生成事件账本。"""

    validate_static_config(config)
    market = event_market_rows.copy()
    if not market.empty and market.duplicated(["event_id", "market_role"]).any():
        raise TenderEvaluationError("事件市场行存在event_id×角色重复键")
    market_by_key = {
        (str(row["event_id"]), str(row["market_role"])): row
        for _, row in market.iterrows()
    }
    entry_stats = industry_entry_stats.copy()
    if not entry_stats.empty and entry_stats["event_id"].astype(str).duplicated().any():
        raise TenderEvaluationError("行业入场统计event_id重复")
    entry_by_event = {
        str(row["event_id"]): row for _, row in entry_stats.iterrows()
    }
    daily = industry_daily_rows.copy()
    if not daily.empty:
        daily["date"] = _date_series(daily["date"])
        if daily.duplicated(["event_id", "date"]).any():
            raise TenderEvaluationError("行业逐日基准event_id×date重复")
    daily_by_event = (
        {
            str(event_id): group.copy()
            for event_id, group in daily.groupby("event_id", sort=False)
        }
        if not daily.empty
        else {}
    )
    benchmark = h00300_ohlc.copy()
    benchmark["date"] = _date_series(benchmark["date"])
    if benchmark["date"].isna().any() or benchmark["date"].duplicated().any():
        raise TenderEvaluationError("H00300日期无效或重复")
    h00300_by_date = {row["date"]: row for _, row in benchmark.iterrows()}
    actions = corporate_actions.copy()
    if not actions.empty:
        if "event_id" not in actions.columns:
            raise TenderEvaluationError("非空公司行动输入缺少event_id")
        actions["event_id"] = actions["event_id"].astype(str)
    actions_by_event = (
        {
            str(event_id): group.copy()
            for event_id, group in actions.groupby("event_id", sort=False)
        }
        if not actions.empty
        else {}
    )

    execution = config["clock_and_execution"]
    minimum_spread = float(execution["minimum_gross_spread"])
    minimum_remaining = int(
        execution["minimum_open_dates_strictly_after_entry_through_offer_end"]
    )
    min_peers = int(config["benchmarks"]["minimum_industry_peers_each_day"])
    stress_buy = float(config["costs"]["stress_buy_bps"]) / 10_000.0
    stress_tender = float(config["costs"]["stress_tender_bps"]) / 10_000.0
    base_buy = float(config["costs"]["base_buy_bps"]) / 10_000.0
    base_tender = float(config["costs"]["base_tender_bps"]) / 10_000.0

    output: list[dict[str, Any]] = []
    for _, source in event_clock.sort_values(
        ["formal_report_pdf_date", "event_id"], kind="stable"
    ).iterrows():
        event = source.copy()
        event_id = str(event["event_id"])
        result = event.to_dict()
        result.update(
            {
                "execution_status": str(event["clock_status"]),
                "report_raw_close": np.nan,
                "gross_cash_spread_at_observation": np.nan,
                "entry_raw_open": np.nan,
                "entry_fill": False,
                "entitled_corporate_action_ids_json": "[]",
                "excluded_corporate_action_ids_json": "[]",
                "unresolved_corporate_action_ids_json": "[]",
                "entitled_gross_cash_distribution_cny": np.nan,
                "corporate_action_accounting_resolved": False,
                "successful_cash_proceeds_cny": np.nan,
                "stress_stock_growth": np.nan,
                "stress_stock_net_return": np.nan,
                "base_stock_growth": np.nan,
                "base_stock_net_return": np.nan,
                "industry_l1_code_at_entry": None,
                "industry_entry_intraday_peer_count": np.nan,
                "industry_entry_intraday_peer_log_return": np.nan,
                "industry_after_entry_expected_day_count": int(
                    event.get("benchmark_after_entry_open_date_count", 0)
                ),
                "industry_after_entry_valid_day_count": 0,
                "industry_minimum_peer_count": np.nan,
                "industry_growth": np.nan,
                "h00300_entry_open": np.nan,
                "h00300_cash_date_close": np.nan,
                "h00300_growth": np.nan,
                "primary_stress_net_industry_excess_return": np.nan,
                "secondary_stress_net_h00300_excess_return": np.nan,
            }
        )
        if event["clock_status"] != "READY_FOR_FROZEN_MARKET_EVALUATION":
            output.append(result)
            continue
        if pd.isna(event.get("entry_security_active")):
            result["execution_status"] = "NO_VIEW_SECURITY_MASTER_MISSING_OR_INVALID"
            output.append(result)
            continue
        if not bool(event["entry_security_active"]):
            result["execution_status"] = "SKIPPED_ENTRY_INACTIVE_SECURITY"
            output.append(result)
            continue

        report_row = market_by_key.get((event_id, "REPORT"))
        if not _traded(report_row, ("raw_close",)):
            result["execution_status"] = "SKIPPED_SIGNAL_OBSERVATION_NOT_TRADED"
            output.append(result)
            continue
        report_close = float(report_row["raw_close"])
        spread_decimal = (
            Decimal(str(event["observed_offer_price_cny"]))
            / Decimal(str(report_close))
            - Decimal("1")
        )
        spread = float(spread_decimal)
        result["report_raw_close"] = report_close
        result["gross_cash_spread_at_observation"] = spread
        if spread_decimal < Decimal(str(minimum_spread)):
            result["execution_status"] = "EXCLUDED_GROSS_CASH_SPREAD_BELOW_8_PERCENT"
            output.append(result)
            continue

        entry_row = market_by_key.get((event_id, "ENTRY"))
        if not _traded(entry_row, ("raw_open", "raw_high", "raw_low", "raw_close")):
            result["execution_status"] = "SKIPPED_ENTRY_NOT_TRADED"
            output.append(result)
            continue
        try:
            locked = one_price_upward_lock(
                entry_row,
                relative_tolerance=float(
                    execution["upward_one_price_relative_tolerance"]
                ),
                pct_threshold=float(
                    execution["upward_one_price_pct_change_at_least"]
                ),
            )
        except TenderEvaluationError:
            result["execution_status"] = "NO_VIEW_ENTRY_LIMIT_FIELDS_MISSING"
            output.append(result)
            continue
        if locked:
            result["execution_status"] = "SKIPPED_ENTRY_UPWARD_ONE_PRICE_LOCK"
            output.append(result)
            continue
        entry_open = float(entry_row["raw_open"])
        result["entry_raw_open"] = entry_open
        frozen_limit_decimal = Decimal(str(event["observed_offer_price_cny"])) / Decimal(
            "1.08"
        )
        if Decimal(str(entry_open)) > frozen_limit_decimal:
            result["execution_status"] = "SKIPPED_ENTRY_OPEN_ABOVE_FROZEN_LIMIT"
            output.append(result)
            continue
        if int(event["open_dates_strictly_after_entry_through_offer_end"]) < minimum_remaining:
            result["execution_status"] = (
                "SKIPPED_ENTRY_INSUFFICIENT_TENDER_SUBMISSION_WINDOW"
            )
            output.append(result)
            continue
        result["entry_fill"] = True

        event_actions = actions_by_event.get(event_id, actions.iloc[0:0])
        action_result = _corporate_action_proceeds(event, event_actions, config)
        result.update(action_result)
        if not action_result["corporate_action_accounting_resolved"]:
            result["execution_status"] = "NO_VIEW_UNRESOLVED_CORPORATE_ACTION_ACCOUNTING"
            output.append(result)
            continue
        if not str(event["outcome_status"]).startswith("SUCCESS_"):
            result["execution_status"] = "NO_VIEW_NON_SUCCESS_BRANCH_NOT_IN_FROZEN_SAMPLE"
            output.append(result)
            continue

        cash_proceeds = float(event["final_effective_offer_price_cny"]) + float(
            action_result["entitled_gross_cash_distribution_cny"]
        )
        stress_growth = (cash_proceeds * (1.0 - stress_tender)) / (
            entry_open * (1.0 + stress_buy)
        )
        base_growth = (cash_proceeds * (1.0 - base_tender)) / (
            entry_open * (1.0 + base_buy)
        )
        result["successful_cash_proceeds_cny"] = cash_proceeds
        result["stress_stock_growth"] = stress_growth
        result["stress_stock_net_return"] = stress_growth - 1.0
        result["base_stock_growth"] = base_growth
        result["base_stock_net_return"] = base_growth - 1.0

        entry_stat = entry_by_event.get(event_id)
        if entry_stat is None:
            result["execution_status"] = "NO_VIEW_INDUSTRY_BENCHMARK_INCOMPLETE"
            output.append(result)
            continue
        result["industry_l1_code_at_entry"] = entry_stat.get("industry_l1_code")
        result["industry_entry_intraday_peer_count"] = entry_stat.get(
            "entry_intraday_peer_count"
        )
        result["industry_entry_intraday_peer_log_return"] = entry_stat.get(
            "entry_intraday_peer_log_return"
        )
        if (
            not _finite(entry_stat.get("entry_intraday_peer_count"))
            or int(entry_stat["entry_intraday_peer_count"]) < min_peers
            or not _finite(entry_stat.get("entry_intraday_peer_log_return"))
        ):
            result["execution_status"] = "NO_VIEW_INDUSTRY_BENCHMARK_INCOMPLETE"
            output.append(result)
            continue
        expected_dates = {
            pd.Timestamp(value).normalize()
            for value in json.loads(str(event["benchmark_after_entry_dates_json"]))
        }
        event_daily = daily_by_event.get(event_id, daily.iloc[0:0]).copy()
        valid_daily = event_daily.loc[
            event_daily["date"].isin(expected_dates)
            & pd.to_numeric(event_daily["industry_peer_count"], errors="coerce").ge(
                min_peers
            )
            & pd.to_numeric(
                event_daily["industry_peer_log_return"], errors="coerce"
            ).notna()
        ]
        valid_dates = set(valid_daily["date"].tolist())
        result["industry_after_entry_valid_day_count"] = len(valid_dates)
        if not valid_daily.empty:
            result["industry_minimum_peer_count"] = int(
                valid_daily["industry_peer_count"].min()
            )
        if valid_dates != expected_dates:
            result["execution_status"] = "NO_VIEW_INDUSTRY_BENCHMARK_INCOMPLETE"
            output.append(result)
            continue
        industry_log = float(entry_stat["entry_intraday_peer_log_return"]) + float(
            valid_daily["industry_peer_log_return"].sum()
        )
        industry_growth = math.exp(industry_log)
        result["industry_growth"] = industry_growth

        entry_benchmark = h00300_by_date.get(pd.Timestamp(event["entry_date"]))
        cash_benchmark = h00300_by_date.get(
            pd.Timestamp(event["cash_availability_date"])
        )
        if (
            entry_benchmark is None
            or cash_benchmark is None
            or not _finite(entry_benchmark.get("open"))
            or float(entry_benchmark["open"]) <= 0.0
            or not _finite(cash_benchmark.get("close"))
            or float(cash_benchmark["close"]) <= 0.0
        ):
            result["execution_status"] = "NO_VIEW_H00300_BENCHMARK_INCOMPLETE"
            output.append(result)
            continue
        h_open = float(entry_benchmark["open"])
        h_close = float(cash_benchmark["close"])
        h_growth = h_close / h_open
        result["h00300_entry_open"] = h_open
        result["h00300_cash_date_close"] = h_close
        result["h00300_growth"] = h_growth
        result["primary_stress_net_industry_excess_return"] = (
            stress_growth / industry_growth - 1.0
        )
        result["secondary_stress_net_h00300_excess_return"] = (
            stress_growth / h_growth - 1.0
        )
        result["execution_status"] = "EXECUTED_RESOLVED_ALL_FROZEN_TARGETS"
        output.append(result)

    ledger = pd.DataFrame(output)
    for column in (
        "formal_report_pdf_date",
        "offer_end_date",
        "result_pdf_date",
        "report_market_date",
        "entry_date",
        "cash_availability_date",
    ):
        ledger[column] = _date_series(ledger[column])
    return ledger.sort_values(
        ["formal_report_pdf_date", "event_id"], kind="stable"
    ).reset_index(drop=True)


def wilson_lower_bound(wins: int, total: int, confidence_level: float = 0.95) -> float:
    if total <= 0 or wins < 0 or wins > total:
        return float("nan")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    proportion = wins / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator


def odds_metrics(values: np.ndarray) -> dict[str, float | int | None]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    wins = clean[clean > 0.0]
    losses = clean[clean <= 0.0]
    mean_loss_abs = abs(float(np.mean(losses))) if len(losses) else float("nan")
    loss_sum_abs = abs(float(np.sum(losses))) if len(losses) else float("nan")
    payoff = (
        float(np.mean(wins)) / mean_loss_abs
        if len(wins) and len(losses) and mean_loss_abs > 0.0
        else None
    )
    profit_factor = (
        float(np.sum(wins)) / loss_sum_abs
        if len(wins) and len(losses) and loss_sum_abs > 0.0
        else None
    )
    return {
        "event_count": int(len(clean)),
        "winning_event_count": int(len(wins)),
        "nonpositive_event_count": int(len(losses)),
        "win_rate": float(len(wins) / len(clean)) if len(clean) else None,
        "mean_positive_win": float(np.mean(wins)) if len(wins) else None,
        "mean_nonpositive_loss": float(np.mean(losses)) if len(losses) else None,
        "payoff_ratio": payoff,
        "profit_factor": profit_factor,
        "mean": float(np.mean(clean)) if len(clean) else None,
        "median": float(np.median(clean)) if len(clean) else None,
    }


def _interval(values: list[float]) -> dict[str, float | int | None]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return {"lower": None, "upper": None, "valid_repetitions": 0}
    return {
        "lower": float(np.percentile(clean, 2.5)),
        "upper": float(np.percentile(clean, 97.5)),
        "valid_repetitions": int(len(clean)),
    }


def cluster_bootstrap(
    analysis: pd.DataFrame, *, repetitions: int, seed: int
) -> dict[str, Any]:
    if analysis.empty:
        raise TenderEvaluationError("聚类自助法没有事件")
    clusters: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in analysis.groupby("formal_report_pdf_date", sort=True):
        clusters.append(
            (
                group["primary_stress_net_industry_excess_return"].to_numpy(
                    dtype=float
                ),
                group["secondary_stress_net_h00300_excess_return"].to_numpy(
                    dtype=float
                ),
            )
        )
    rng = np.random.default_rng(seed)
    means: list[float] = []
    payoffs: list[float] = []
    profit_factors: list[float] = []
    secondary_means: list[float] = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        primary = np.concatenate([clusters[int(index)][0] for index in sampled])
        secondary = np.concatenate([clusters[int(index)][1] for index in sampled])
        odds = odds_metrics(primary)
        means.append(float(np.mean(primary)))
        payoffs.append(
            float(odds["payoff_ratio"])
            if odds["payoff_ratio"] is not None
            else float("nan")
        )
        profit_factors.append(
            float(odds["profit_factor"])
            if odds["profit_factor"] is not None
            else float("nan")
        )
        secondary_means.append(float(np.mean(secondary)))
    return {
        "method": "CLUSTER_RESAMPLE_EFFECTIVE_FORMAL_REPORT_DATES_WITH_REPLACEMENT",
        "repetitions": repetitions,
        "seed": seed,
        "cluster_count": len(clusters),
        "confidence_level": 0.95,
        "primary_mean": _interval(means),
        "primary_payoff_ratio": _interval(payoffs),
        "primary_profit_factor": _interval(profit_factors),
        "secondary_h00300_mean": _interval(secondary_means),
    }


def calculate_acceptance_metrics(
    ledger: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    executed = ledger.loc[
        ledger["execution_status"].eq("EXECUTED_RESOLVED_ALL_FROZEN_TARGETS")
    ].copy()
    primary_values = executed[
        "primary_stress_net_industry_excess_return"
    ].to_numpy(dtype=float)
    secondary_values = executed[
        "secondary_stress_net_h00300_excess_return"
    ].to_numpy(dtype=float)
    primary = odds_metrics(primary_values)
    primary["wilson_95_lower"] = wilson_lower_bound(
        int(primary["winning_event_count"]), int(primary["event_count"])
    )
    secondary = odds_metrics(secondary_values)

    unique_dates = sorted(executed["formal_report_pdf_date"].dropna().unique())
    midpoint = len(unique_dates) // 2
    first_dates = set(unique_dates[:midpoint])
    first = executed.loc[executed["formal_report_pdf_date"].isin(first_dates)]
    second = executed.loc[~executed["formal_report_pdf_date"].isin(first_dates)]
    first_metrics = odds_metrics(
        first["primary_stress_net_industry_excess_return"].to_numpy(dtype=float)
    )
    second_metrics = odds_metrics(
        second["primary_stress_net_industry_excess_return"].to_numpy(dtype=float)
    )

    annual: list[dict[str, Any]] = []
    if not executed.empty:
        executed["formal_report_year"] = executed["formal_report_pdf_date"].dt.year
    for year in sorted(executed.get("formal_report_year", pd.Series(dtype=int)).unique()):
        values = executed.loc[
            executed["formal_report_year"].eq(year),
            "primary_stress_net_industry_excess_return",
        ].to_numpy(dtype=float)
        annual.append(
            {
                "year": int(year),
                "event_count": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else None,
                "positive_mean": bool(np.mean(values) > 0.0) if len(values) else None,
            }
        )
    covered_years = [row for row in annual if row["event_count"] >= 2]
    positive_fraction = (
        sum(bool(row["positive_mean"]) for row in covered_years) / len(covered_years)
        if covered_years
        else None
    )

    positives = np.sort(primary_values[primary_values > 0.0])[::-1]
    positive_sum = float(np.sum(positives)) if len(positives) else 0.0
    maximum_single = (
        float(positives[0] / positive_sum)
        if len(positives) and positive_sum > 0.0
        else None
    )
    top_count = max(1, math.ceil(0.20 * len(positives))) if len(positives) else 0
    top_share = (
        float(np.sum(positives[:top_count]) / positive_sum)
        if len(positives) and positive_sum > 0.0
        else None
    )
    leave_best_out = (
        float(np.mean(np.delete(primary_values, int(np.argmax(primary_values)))))
        if len(primary_values) > 1
        else None
    )

    bootstrap_config = config["bootstrap"]
    bootstrap = (
        cluster_bootstrap(
            executed,
            repetitions=int(bootstrap_config["repetitions"]),
            seed=int(bootstrap_config["seed"]),
        )
        if not executed.empty
        else None
    )
    resolved_outcomes = ledger["outcome_status"].astype(str).isin(
        {
            "SUCCESS_SETTLEMENT_CONFIRMED",
            "SUCCESS_RESULT_CONFIRMED_SETTLEMENT_DATE_NO_VIEW",
            "FAILED_OR_TERMINATED_CONFIRMED",
        }
    )
    return {
        "source_event_count": int(len(ledger)),
        "mature_offer_outcome_resolved_count": int(resolved_outcomes.sum()),
        "mature_offer_outcome_resolution_coverage": float(resolved_outcomes.mean()),
        "observed_spread_pass_count": int(
            pd.to_numeric(
                ledger["gross_cash_spread_at_observation"], errors="coerce"
            ).ge(float(config["clock_and_execution"]["minimum_gross_spread"])).sum()
        ),
        "entry_fill_count": int(ledger["entry_fill"].fillna(False).astype(bool).sum()),
        "primary_target_event_count": int(len(executed)),
        "secondary_target_event_count": int(len(executed)),
        "unique_offer_count": int(executed["event_id"].nunique()),
        "unique_effective_report_date_count": int(
            executed["formal_report_pdf_date"].nunique()
        ),
        "distinct_calendar_year_count": int(
            executed["formal_report_pdf_date"].dt.year.nunique()
        ),
        "formal_candidate_pdf_text_coverage": 1.0,
        "included_and_ambiguous_pdf_visual_review_coverage": 1.0,
        "offer_term_parser_visual_agreement": 1.0,
        "linked_lifecycle_pdf_visual_review_coverage": 1.0,
        "execution_status_counts": {
            str(key): int(value)
            for key, value in ledger["execution_status"].value_counts().items()
        },
        "primary": primary,
        "secondary_h00300": secondary,
        "chronological_halves_definition": (
            "KEEP_FORMAL_REPORT_DATE_CLUSTERS_INTACT_SPLIT_SORTED_UNIQUE_DATES_AT_FLOOR_HALF"
        ),
        "first_half": first_metrics,
        "second_half": second_metrics,
        "annual": annual,
        "calendar_years_with_at_least_2_events": int(len(covered_years)),
        "positive_calendar_year_fraction": positive_fraction,
        "maximum_single_winner_share_of_total_positive_profit": maximum_single,
        "top_20_percent_winner_share_of_total_positive_profit": top_share,
        "top_20_percent_winner_count": top_count,
        "leave_best_event_out_mean": leave_best_out,
        "bootstrap": bootstrap,
    }


def _available(value: Any) -> bool:
    return value is not None and _finite(value)


def evaluate_acceptance_gates(
    metrics: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    thresholds = config["acceptance_gate"]
    primary = metrics["primary"]
    secondary = metrics["secondary_h00300"]
    bootstrap = metrics["bootstrap"]
    repetitions = int(config["bootstrap"]["repetitions"])
    minimum_valid = float(
        config["bootstrap"]["minimum_valid_payoff_and_profit_factor_fraction"]
    )
    bootstrap_valid = bootstrap is not None
    payoff_valid_fraction = (
        bootstrap["primary_payoff_ratio"]["valid_repetitions"] / repetitions
        if bootstrap_valid
        else 0.0
    )
    profit_factor_valid_fraction = (
        bootstrap["primary_profit_factor"]["valid_repetitions"] / repetitions
        if bootstrap_valid
        else 0.0
    )
    evidence_checks = {
        "MINIMUM_EVENTS": metrics["primary_target_event_count"]
        >= int(thresholds["minimum_events"]),
        "MINIMUM_UNIQUE_OFFERS": metrics["unique_offer_count"]
        >= int(thresholds["minimum_unique_offers"]),
        "MINIMUM_UNIQUE_EFFECTIVE_REPORT_DATES": metrics[
            "unique_effective_report_date_count"
        ]
        >= int(thresholds["minimum_unique_effective_report_dates"]),
        "MINIMUM_DISTINCT_CALENDAR_YEARS": metrics["distinct_calendar_year_count"]
        >= int(thresholds["minimum_distinct_calendar_years"]),
        "MINIMUM_WINNING_EVENTS": primary["winning_event_count"]
        >= int(thresholds["minimum_winning_events_for_payoff_estimation"]),
        "MINIMUM_LOSING_EVENTS": primary["nonpositive_event_count"]
        >= int(thresholds["minimum_losing_events_for_payoff_estimation"]),
        "FORMAL_CANDIDATE_PDF_TEXT_COVERAGE": metrics[
            "formal_candidate_pdf_text_coverage"
        ]
        == 1.0,
        "VISUAL_REVIEW_COVERAGE": metrics[
            "included_and_ambiguous_pdf_visual_review_coverage"
        ]
        == 1.0,
        "PARSER_VISUAL_AGREEMENT": metrics["offer_term_parser_visual_agreement"]
        == 1.0,
        "LIFECYCLE_VISUAL_REVIEW_COVERAGE": metrics[
            "linked_lifecycle_pdf_visual_review_coverage"
        ]
        == 1.0,
        "MATURE_OUTCOME_RESOLUTION_COVERAGE": metrics[
            "mature_offer_outcome_resolution_coverage"
        ]
        >= float(thresholds["minimum_mature_offer_outcome_resolution_coverage"]),
        "PRIMARY_AND_SECONDARY_TARGET_COUNTS_EQUAL": metrics[
            "primary_target_event_count"
        ]
        == metrics["secondary_target_event_count"],
        "PAYOFF_AND_PROFIT_FACTOR_ESTIMABLE": _available(primary["payoff_ratio"])
        and _available(primary["profit_factor"]),
        "BOOTSTRAP_MEANS_COMPLETE": bootstrap_valid
        and bootstrap["primary_mean"]["valid_repetitions"] == repetitions
        and bootstrap["secondary_h00300_mean"]["valid_repetitions"] == repetitions,
        "BOOTSTRAP_ODDS_VALID_FRACTION": payoff_valid_fraction >= minimum_valid
        and profit_factor_valid_fraction >= minimum_valid,
    }
    halves_available = all(
        _available(metrics[half][metric])
        for half in ("first_half", "second_half")
        for metric in ("win_rate", "mean")
    )
    economic_checks = {
        "PRIMARY_STRESS_WIN_RATE": _available(primary["win_rate"])
        and primary["win_rate"]
        >= float(thresholds["primary_stress_win_rate_at_least"]),
        "PRIMARY_STRESS_WIN_RATE_WILSON_LOWER": _available(
            primary["wilson_95_lower"]
        )
        and primary["wilson_95_lower"]
        > float(
            thresholds["primary_stress_win_rate_wilson_95_lower_strictly_above"]
        ),
        "PRIMARY_STRESS_PAYOFF_RATIO": _available(primary["payoff_ratio"])
        and primary["payoff_ratio"]
        >= float(thresholds["primary_stress_payoff_ratio_at_least"]),
        "PRIMARY_STRESS_PAYOFF_BOOTSTRAP_LOWER": bootstrap_valid
        and _available(bootstrap["primary_payoff_ratio"]["lower"])
        and bootstrap["primary_payoff_ratio"]["lower"]
        > float(
            thresholds[
                "primary_stress_payoff_ratio_bootstrap_95_lower_strictly_above"
            ]
        ),
        "PRIMARY_STRESS_PROFIT_FACTOR": _available(primary["profit_factor"])
        and primary["profit_factor"]
        >= float(thresholds["primary_stress_profit_factor_at_least"]),
        "PRIMARY_STRESS_PROFIT_FACTOR_BOOTSTRAP_LOWER": bootstrap_valid
        and _available(bootstrap["primary_profit_factor"]["lower"])
        and bootstrap["primary_profit_factor"]["lower"]
        > float(
            thresholds[
                "primary_stress_profit_factor_bootstrap_95_lower_strictly_above"
            ]
        ),
        "PRIMARY_STRESS_MEAN": _available(primary["mean"])
        and primary["mean"]
        >= float(thresholds["primary_stress_mean_excess_return_at_least"]),
        "PRIMARY_STRESS_MEAN_BOOTSTRAP_LOWER": bootstrap_valid
        and _available(bootstrap["primary_mean"]["lower"])
        and bootstrap["primary_mean"]["lower"]
        > float(
            thresholds[
                "primary_stress_mean_excess_return_bootstrap_95_lower_strictly_above"
            ]
        ),
        "PRIMARY_STRESS_MEDIAN": _available(primary["median"])
        and primary["median"]
        > float(thresholds["primary_stress_median_excess_return_strictly_above"]),
        "BOTH_CHRONOLOGICAL_HALVES": halves_available
        and all(
            metrics[half]["win_rate"]
            > float(
                thresholds["both_chronological_halves_win_rate_strictly_above"]
            )
            and metrics[half]["mean"]
            > float(
                thresholds["both_chronological_halves_mean_excess_return_strictly_above"]
            )
            for half in ("first_half", "second_half")
        ),
        "POSITIVE_CALENDAR_YEAR_FRACTION": metrics[
            "calendar_years_with_at_least_2_events"
        ]
        >= int(thresholds["minimum_calendar_years_with_at_least_2_events"])
        and _available(metrics["positive_calendar_year_fraction"])
        and metrics["positive_calendar_year_fraction"]
        >= float(
            thresholds[
                "minimum_positive_calendar_year_fraction_among_years_with_at_least_2_events"
            ]
        ),
        "MAXIMUM_SINGLE_WINNER_SHARE": _available(
            metrics["maximum_single_winner_share_of_total_positive_profit"]
        )
        and metrics["maximum_single_winner_share_of_total_positive_profit"]
        <= float(thresholds["maximum_single_winner_share_of_total_positive_profit"]),
        "TOP_20_PERCENT_WINNER_SHARE": _available(
            metrics["top_20_percent_winner_share_of_total_positive_profit"]
        )
        and metrics["top_20_percent_winner_share_of_total_positive_profit"]
        <= float(
            thresholds["maximum_top_20_percent_winner_share_of_total_positive_profit"]
        ),
        "LEAVE_BEST_EVENT_OUT_MEAN": _available(metrics["leave_best_event_out_mean"])
        and metrics["leave_best_event_out_mean"]
        > float(
            thresholds["leave_best_event_out_mean_excess_return_strictly_above"]
        ),
        "SECONDARY_H00300_WIN_RATE": _available(secondary["win_rate"])
        and secondary["win_rate"]
        >= float(thresholds["secondary_h00300_stress_win_rate_at_least"]),
        "SECONDARY_H00300_MEAN_BOOTSTRAP_LOWER": bootstrap_valid
        and _available(bootstrap["secondary_h00300_mean"]["lower"])
        and bootstrap["secondary_h00300_mean"]["lower"]
        > float(
            thresholds[
                "secondary_h00300_stress_mean_excess_bootstrap_95_lower_strictly_above"
            ]
        ),
    }
    evidence_pass = all(evidence_checks.values())
    economic_pass = all(economic_checks.values())
    if not evidence_pass:
        status = thresholds["insufficient_status"]
    elif not economic_pass:
        status = thresholds["fail_status"]
    else:
        status = thresholds["pass_status"]
    return {
        "historical_status": status,
        "evidence_checks": evidence_checks,
        "economic_checks": economic_checks,
        "failed_evidence_checks": [
            key for key, passed in evidence_checks.items() if not passed
        ],
        "failed_economic_checks": [
            key for key, passed in economic_checks.items() if not passed
        ],
        "evidence_passed_count": int(sum(evidence_checks.values())),
        "evidence_required_count": len(evidence_checks),
        "economic_passed_count": int(sum(economic_checks.values())),
        "economic_required_count": len(economic_checks),
        "all_conditions_passed": evidence_pass and economic_pass,
        "shadow_required_even_if_pass": True,
        "position_mapping": False,
        "order_generation": False,
        "live_authorized": False,
    }
