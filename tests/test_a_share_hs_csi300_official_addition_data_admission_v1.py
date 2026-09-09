from __future__ import annotations

import math

import pytest

from research.a_share_hs_csi300_official_addition_data_admission_v1 import (
    applicable_price_limit_rule,
    classify_entry_execution,
    classify_sellability,
    component_gate,
    rounded_limit_price,
)


PRICE_LIMIT_RULES = {
    "main_board_normal_fraction": 0.10,
    "main_board_st_fraction": 0.05,
    "star_board_fraction": 0.20,
    "chinext_fraction_before_2020_08_24_normal": 0.10,
    "chinext_fraction_before_2020_08_24_st": 0.05,
    "chinext_fraction_from_2020_08_24": 0.20,
}


def price_limit_rule(
    ts_code: str,
    *,
    trade_date: str = "2026-06-01",
    security_status: str = "NORMAL",
    listing_trade_day_number: int = 100,
    list_date: str = "2010-01-01",
):
    return applicable_price_limit_rule(
        ts_code=ts_code,
        trade_date=trade_date,
        security_status=security_status,
        listing_trade_day_number=listing_trade_day_number,
        list_date=list_date,
        rules=PRICE_LIMIT_RULES,
    )


@pytest.mark.parametrize(
    ("ts_code", "trade_date", "security_status", "expected_board", "expected_fraction"),
    [
        ("600001.SH", "2026-06-01", "NORMAL", "SSE_MAIN", 0.10),
        ("000001.SZ", "2026-06-01", "ST", "SZSE_MAIN", 0.05),
        ("688001.SH", "2026-06-01", "NORMAL", "SSE_STAR", 0.20),
        ("300001.SZ", "2020-08-21", "NORMAL", "SZSE_CHINEXT", 0.10),
        ("300001.SZ", "2020-08-24", "NORMAL", "SZSE_CHINEXT", 0.20),
    ],
)
def test_price_limit_rule_by_board_and_status(
    ts_code: str,
    trade_date: str,
    security_status: str,
    expected_board: str,
    expected_fraction: float,
) -> None:
    rule = price_limit_rule(
        ts_code,
        trade_date=trade_date,
        security_status=security_status,
    )
    assert rule.board == expected_board
    assert rule.status == "PRICE_LIMIT_RULE_RESOLVED"
    assert rule.fraction == pytest.approx(expected_fraction)


@pytest.mark.parametrize(
    ("ts_code", "list_date"),
    [
        ("688001.SH", "2019-07-22"),
        ("300001.SZ", "2020-08-24"),
        ("600001.SH", "2023-04-10"),
    ],
)
def test_modern_first_five_listing_days_have_no_daily_limit(
    ts_code: str,
    list_date: str,
) -> None:
    rule = price_limit_rule(
        ts_code,
        trade_date=list_date,
        listing_trade_day_number=1,
        list_date=list_date,
    )
    assert rule.status == "NO_DAILY_LIMIT_FIRST_FIVE_LISTING_DAYS"
    assert rule.fraction is None


def test_legacy_ipo_first_day_is_no_view() -> None:
    rule = price_limit_rule(
        "600001.SH",
        trade_date="2020-01-02",
        listing_trade_day_number=1,
        list_date="2020-01-02",
    )
    assert rule.status == "NO_VIEW_LEGACY_IPO_FIRST_DAY_RULE_UNRESOLVED"
    assert rule.fraction is None


def test_exchange_price_limit_rounding_uses_half_up() -> None:
    assert rounded_limit_price(10.05, 0.10, upper=True) == 11.06
    assert rounded_limit_price(10.05, 0.10, upper=False) == 9.05
    assert math.isnan(rounded_limit_price(float("nan"), 0.10, upper=True))


def test_entry_execution_handles_suspension_and_one_price_limit_up() -> None:
    rule = price_limit_rule("600001.SH")
    suspended = {
        "market_row_present": True,
        "is_suspended": True,
    }
    one_price_limit_up = {
        "market_row_present": True,
        "is_suspended": False,
        "pre_close": 10.05,
        "raw_open": 11.06,
        "raw_high": 11.06,
        "raw_low": 11.06,
        "raw_close": 11.06,
        "volume": 1000,
    }
    ordinary_open = {
        **one_price_limit_up,
        "raw_open": 10.10,
        "raw_high": 10.30,
        "raw_low": 10.00,
        "raw_close": 10.20,
    }
    assert classify_entry_execution(suspended, rule) == "NO_FILL_SUSPENDED"
    assert classify_entry_execution(one_price_limit_up, rule) == "NO_FILL_ONE_PRICE_LIMIT_UP"
    assert classify_entry_execution(ordinary_open, rule) == "FILL_AT_RAW_OPEN"


def test_exit_sellability_handles_effective_and_delayed_clocks() -> None:
    rule = price_limit_rule("600001.SH")
    ordinary = {
        "market_row_present": True,
        "is_suspended": False,
        "pre_close": 10.00,
        "raw_open": 9.95,
        "raw_high": 10.10,
        "raw_low": 9.90,
        "raw_close": 10.05,
        "volume": 1000,
    }
    suspended = {
        "market_row_present": True,
        "is_suspended": True,
    }
    one_price_limit_down = {
        "market_row_present": True,
        "is_suspended": False,
        "pre_close": 10.05,
        "raw_open": 9.05,
        "raw_high": 9.05,
        "raw_low": 9.05,
        "raw_close": 9.05,
        "volume": 1000,
    }
    assert classify_sellability(ordinary, rule, phase="EFFECTIVE_CLOSE") == "SELLABLE_EFFECTIVE_CLOSE"
    assert classify_sellability(suspended, rule, phase="EFFECTIVE_CLOSE") == "BLOCKED_EFFECTIVE_SUSPENDED"
    assert (
        classify_sellability(one_price_limit_down, rule, phase="EFFECTIVE_CLOSE")
        == "BLOCKED_EFFECTIVE_ONE_PRICE_LIMIT_DOWN"
    )
    assert classify_sellability(ordinary, rule, phase="DELAYED_OPEN") == "SELLABLE_DELAYED_OPEN"


def test_d4_component_gate_requires_every_component() -> None:
    passed, failures = component_gate(
        {"行情": 0.999, "公司行为": 0.9799},
        0.98,
    )
    assert passed is False
    assert failures == ["公司行为"]

    passed, failures = component_gate(
        {"行情": 0.98, "公司行为": 1.0},
        0.98,
    )
    assert passed is True
    assert failures == []
