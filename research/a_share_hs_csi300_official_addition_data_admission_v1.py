"""沪深300官方调入候选 D4/D5 数据准入纯函数。

本模块只生成覆盖布尔值、状态代码和日期，不输出任何价格或收益数值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class PriceLimitRule:
    board: str
    fraction: float | None
    status: str


def finite_positive(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def finite_nonnegative(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0.0


def classify_board(ts_code: str) -> str:
    code = str(ts_code).upper().split(".", 1)[0]
    suffix = str(ts_code).upper().split(".", 1)[-1] if "." in str(ts_code) else ""
    if suffix == "SH" and code.startswith(("688", "689")):
        return "SSE_STAR"
    if suffix == "SZ" and code.startswith(("300", "301")):
        return "SZSE_CHINEXT"
    if suffix == "SH":
        return "SSE_MAIN"
    if suffix == "SZ":
        return "SZSE_MAIN"
    return "UNKNOWN_BOARD"


def exchange_price_round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float("nan")
    return float(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def applicable_price_limit_rule(
    *,
    ts_code: str,
    trade_date: str | date | pd.Timestamp,
    security_status: str | None,
    listing_trade_day_number: int | None,
    list_date: str | date | pd.Timestamp | None,
    rules: Mapping[str, Any],
) -> PriceLimitRule:
    board = classify_board(ts_code)
    if board == "UNKNOWN_BOARD":
        return PriceLimitRule(board, None, "NO_VIEW_UNKNOWN_BOARD")
    if security_status is None or pd.isna(security_status):
        return PriceLimitRule(board, None, "NO_VIEW_SECURITY_STATUS_MISSING")
    status = str(security_status).upper().replace(" ", "")
    if "DELIST" in status or "TERMINATED" in status or "退" in status:
        return PriceLimitRule(board, None, "NO_VIEW_DELISTING_OR_TERMINATED")
    is_st = "ST" in status
    trading_day = pd.Timestamp(trade_date).normalize()
    listed = None if list_date is None or pd.isna(list_date) else pd.Timestamp(list_date).normalize()
    listing_day = None if listing_trade_day_number is None or pd.isna(listing_trade_day_number) else int(listing_trade_day_number)

    if listing_day is not None and listing_day <= 5:
        modern_no_limit = (
            board == "SSE_STAR"
            or (board == "SZSE_CHINEXT" and listed is not None and listed >= pd.Timestamp("2020-08-24"))
            or (
                board in {"SSE_MAIN", "SZSE_MAIN"}
                and listed is not None
                and listed >= pd.Timestamp("2023-04-10")
            )
        )
        if modern_no_limit:
            return PriceLimitRule(board, None, "NO_DAILY_LIMIT_FIRST_FIVE_LISTING_DAYS")
        if listing_day == 1:
            return PriceLimitRule(board, None, "NO_VIEW_LEGACY_IPO_FIRST_DAY_RULE_UNRESOLVED")

    if board in {"SSE_MAIN", "SZSE_MAIN"}:
        fraction = rules["main_board_st_fraction"] if is_st else rules["main_board_normal_fraction"]
        return PriceLimitRule(board, float(fraction), "PRICE_LIMIT_RULE_RESOLVED")
    if board == "SSE_STAR":
        return PriceLimitRule(board, float(rules["star_board_fraction"]), "PRICE_LIMIT_RULE_RESOLVED")
    if board == "SZSE_CHINEXT":
        if trading_day >= pd.Timestamp("2020-08-24"):
            return PriceLimitRule(board, float(rules["chinext_fraction_from_2020_08_24"]), "PRICE_LIMIT_RULE_RESOLVED")
        key = "chinext_fraction_before_2020_08_24_st" if is_st else "chinext_fraction_before_2020_08_24_normal"
        return PriceLimitRule(board, float(rules[key]), "PRICE_LIMIT_RULE_RESOLVED")
    return PriceLimitRule(board, None, "NO_VIEW_PRICE_LIMIT_RULE_UNRESOLVED")


def rounded_limit_price(pre_close: float, fraction: float, *, upper: bool) -> float:
    if not finite_positive(pre_close):
        return float("nan")
    multiplier = Decimal("1") + (Decimal(str(fraction)) if upper else -Decimal(str(fraction)))
    value = (Decimal(str(float(pre_close))) * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(value)


def at_price_limit(observed_price: float, pre_close: float, fraction: float, *, upper: bool) -> bool:
    if not finite_positive(observed_price) or not finite_positive(pre_close):
        return False
    limit_price = rounded_limit_price(pre_close, fraction, upper=upper)
    tolerance = 0.005000001
    if upper:
        return bool(float(observed_price) >= limit_price - tolerance)
    return bool(float(observed_price) <= limit_price + tolerance)


def is_one_price_day(row: Mapping[str, Any]) -> bool:
    values = [row.get(column) for column in ("raw_open", "raw_high", "raw_low", "raw_close")]
    if not all(finite_positive(value) for value in values):
        return False
    return max(float(value) for value in values) - min(float(value) for value in values) <= 0.005000001


def market_row_component_coverage(row: Mapping[str, Any], rule: PriceLimitRule) -> dict[str, bool]:
    row_present = bool(row.get("market_row_present"))
    suspension_known = row_present and row.get("is_suspended") is not None and not pd.isna(row.get("is_suspended"))
    suspended = bool(row.get("is_suspended")) if suspension_known else False
    open_covered = suspension_known and (suspended or finite_positive(row.get("raw_open")))
    close_covered = suspension_known and (suspended or finite_positive(row.get("raw_close")))
    amount_covered = suspension_known and (suspended or finite_nonnegative(row.get("amount")))
    price_fields = all(finite_positive(row.get(column)) for column in ("pre_close", "raw_open", "raw_high", "raw_low", "raw_close"))
    rule_resolved = rule.status in {"PRICE_LIMIT_RULE_RESOLVED", "NO_DAILY_LIMIT_FIRST_FIVE_LISTING_DAYS"}
    price_limit_covered = suspension_known and (suspended or (price_fields and rule_resolved))
    action_factor_covered = suspension_known and (suspended or finite_positive(row.get("linked_adjustment_factor")))
    return {
        "market_row_covered": row_present,
        "raw_open_covered": open_covered,
        "raw_close_covered": close_covered,
        "amount_covered": amount_covered,
        "suspension_covered": suspension_known,
        "price_limit_covered": price_limit_covered,
        "linked_adjustment_factor_covered": action_factor_covered,
    }


def classify_entry_execution(row: Mapping[str, Any], rule: PriceLimitRule) -> str:
    if not bool(row.get("market_row_present")):
        return "NO_VIEW_ENTRY_MARKET_ROW_MISSING"
    if row.get("is_suspended") is None or pd.isna(row.get("is_suspended")):
        return "NO_VIEW_ENTRY_SUSPENSION_STATUS_MISSING"
    if bool(row.get("is_suspended")):
        return "NO_FILL_SUSPENDED"
    if not finite_positive(row.get("raw_open")) or not finite_positive(row.get("volume")):
        return "NO_VIEW_ENTRY_OPEN_OR_VOLUME_MISSING"
    if rule.status.startswith("NO_VIEW"):
        return rule.status.replace("NO_VIEW_", "NO_VIEW_ENTRY_", 1)
    if rule.fraction is not None:
        if not finite_positive(row.get("pre_close")) or not all(
            finite_positive(row.get(column)) for column in ("raw_high", "raw_low", "raw_close")
        ):
            return "NO_VIEW_ENTRY_PRICE_LIMIT_FIELDS_MISSING"
        if is_one_price_day(row) and at_price_limit(
            float(row["raw_open"]), float(row["pre_close"]), rule.fraction, upper=True
        ):
            return "NO_FILL_ONE_PRICE_LIMIT_UP"
    return "FILL_AT_RAW_OPEN"


def classify_sellability(row: Mapping[str, Any], rule: PriceLimitRule, *, phase: str) -> str:
    prefix = "EFFECTIVE" if phase == "EFFECTIVE_CLOSE" else "DELAYED"
    if not bool(row.get("market_row_present")):
        return f"NO_VIEW_{prefix}_MARKET_ROW_MISSING"
    if row.get("is_suspended") is None or pd.isna(row.get("is_suspended")):
        return f"NO_VIEW_{prefix}_SUSPENSION_STATUS_MISSING"
    if bool(row.get("is_suspended")):
        return f"BLOCKED_{prefix}_SUSPENDED"
    observed_field = "raw_close" if phase == "EFFECTIVE_CLOSE" else "raw_open"
    if not finite_positive(row.get(observed_field)) or not finite_positive(row.get("volume")):
        return f"NO_VIEW_{prefix}_PRICE_OR_VOLUME_MISSING"
    if is_one_price_day(row) and finite_positive(row.get("pre_close")) and float(row["raw_close"]) < float(row["pre_close"]):
        if rule.status.startswith("NO_VIEW"):
            return rule.status.replace("NO_VIEW_", f"NO_VIEW_{prefix}_", 1)
        if rule.fraction is not None and at_price_limit(
            float(row["raw_close"]), float(row["pre_close"]), rule.fraction, upper=False
        ):
            return f"BLOCKED_{prefix}_ONE_PRICE_LIMIT_DOWN"
    return "SELLABLE_EFFECTIVE_CLOSE" if phase == "EFFECTIVE_CLOSE" else "SELLABLE_DELAYED_OPEN"


def component_gate(component_coverages: Mapping[str, float], minimum: float) -> tuple[bool, list[str]]:
    failures = [name for name, value in component_coverages.items() if not math.isfinite(float(value)) or float(value) < minimum]
    return not failures, failures

