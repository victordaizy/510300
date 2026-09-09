"""510300全市场换手率可见度V1的冻结单元测试。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_turnover_visibility_v1 import (  # noqa: E402
    ContractError,
    add_mechanism_targets,
    build_monthly_factor,
    build_monthly_factor_from_components,
    build_signal_schedule,
    circular_block_bootstrap_slope,
    load_config,
    newey_west_slope,
    parse_sse_current_daily,
    parse_sse_current_monthly,
    parse_sse_legacy_daily,
    parse_sse_legacy_monthly,
    parse_szse_daily,
    parse_szse_monthly,
    validate_config,
)
from build_510300_aggregate_turnover_inputs_v1 import (  # noqa: E402
    _parse_sequential_decimal_values,
    parse_szse_static_month,
)


def _jsonp(document: dict) -> bytes:
    return ("jsonpCallback(" + json.dumps(document) + ")").encode("utf-8")


def test_config_locks_external_coefficients_and_scope() -> None:
    config = load_config()
    assert config["external_allocation"]["intercept"] == -0.019
    assert config["external_allocation"]["turnover_coefficient"] == 0.258
    assert config["external_allocation"]["fixed_monthly_standard_deviation"] == 0.0901
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["dates"]["pre_2015_returns_allowed"] is False
    assert config["data_contract"]["szse"]["proxy_scope_is_frozen"] is True
    assert config["data_contract"]["szse"]["transport"] == (
        "HTTPS_ONLY_TLS_VERIFICATION_REQUIRED"
    )
    assert config["data_contract"]["historical_availability_status"] == (
        "HISTORICAL_ARCHIVE_RECONSTRUCTION_NOT_TIMESTAMP_PROOF"
    )
    assert config["protocol"]["pre_freeze_price_input_inspection"][
        "pristine_blind_holdout_claim_allowed"
    ] is False
    assert config["portfolio"]["initial_capital_cny"] == 20000.0
    assert config["portfolio"]["lot_size_shares"] == 100
    assert config["portfolio"]["base_costs"]["minimum_commission_cny_per_leg"] == 5.0
    assert config["portfolio"]["base_costs"]["slippage_bps_per_leg"] == 5.0
    assert config["portfolio"]["stress_costs"]["slippage_bps_per_leg"] == 10.0


def test_config_rejects_coefficient_rescue() -> None:
    config = load_config()
    modified = deepcopy(config)
    modified["external_allocation"]["turnover_coefficient"] = 0.259
    with pytest.raises(ContractError, match="turnover_coefficient"):
        validate_config(modified)


def test_parse_sse_legacy_daily_sums_only_a_and_star() -> None:
    payload = _jsonp(
        {
            "result": [
                {
                    "PRODUCT_TYPE": "1",
                    "CAL_DATE": "2021-12-24 00:00:00.0",
                    "TX_AMOUNT_FULL": "100.25",
                    "MKT_VALUE_FULL": "1000.5",
                },
                {
                    "PRODUCT_TYPE": "2",
                    "CAL_DATE": "2021-12-24 00:00:00.0",
                    "TX_AMOUNT_FULL": "3.0",
                    "MKT_VALUE_FULL": "20.0",
                },
                {
                    "PRODUCT_TYPE": "48",
                    "CAL_DATE": "2021-12-24 00:00:00.0",
                    "TX_AMOUNT_FULL": "20.75",
                    "MKT_VALUE_FULL": "200.5",
                },
            ]
        }
    )
    parsed = parse_sse_legacy_daily(payload, "2021-12-24")
    assert parsed["trade_amount_cny_100m"] == pytest.approx(121.0)
    assert parsed["market_cap_cny_100m"] == pytest.approx(1201.0)
    assert parsed["selected_categories_observed"] == ["1", "48"]


def test_parse_sse_current_daily_sums_main_a_and_star() -> None:
    payload = _jsonp(
        {
            "result": [
                {
                    "PRODUCT_CODE": "01",
                    "TRADE_DATE": "20260731",
                    "TRADE_AMT": "100.1",
                    "TOTAL_VALUE": "1000.2",
                },
                {
                    "PRODUCT_CODE": "02",
                    "TRADE_DATE": "20260731",
                    "TRADE_AMT": "1.0",
                    "TOTAL_VALUE": "10.0",
                },
                {
                    "PRODUCT_CODE": "03",
                    "TRADE_DATE": "20260731",
                    "TRADE_AMT": "25.2",
                    "TOTAL_VALUE": "300.3",
                },
            ]
        }
    )
    parsed = parse_sse_current_daily(payload, "2026-07-31")
    assert parsed["trade_amount_cny_100m"] == pytest.approx(125.3)
    assert parsed["market_cap_cny_100m"] == pytest.approx(1300.5)
    assert parsed["selected_categories_observed"] == ["01", "03"]


def test_parse_szse_daily_requires_exact_date_echo_and_rows() -> None:
    payload = json.dumps(
        [
            {
                "metadata": {
                    "tabkey": "tab1",
                    "conditions": [
                        {"name": "txtQueryDate", "defaultValue": "2026-07-31"}
                    ],
                },
                "data": [
                    {"zbmc": "股票总市值（亿元）", "brsz": "424,903.03"},
                    {"zbmc": "股票成交金额（亿元）", "brsz": "13,554.50"},
                ],
            }
        ]
    ).encode("utf-8")
    parsed = parse_szse_daily(payload, "2026-07-31")
    assert parsed["market_cap_cny_100m"] == pytest.approx(424903.03)
    assert parsed["trade_amount_cny_100m"] == pytest.approx(13554.50)


def test_parse_monthly_crosschecks() -> None:
    legacy = _jsonp(
        {
            "result": [
                {
                    "PRODUCT_TYPE": "1",
                    "TX_AMOUNT": "100",
                    "MKT_VALUE": "1000",
                },
                {
                    "PRODUCT_TYPE": "2",
                    "TX_AMOUNT": "2",
                    "MKT_VALUE": "20",
                },
            ]
        }
    )
    current = _jsonp(
        {
            "result": [
                {
                    "PRODUCT_CODE": "01",
                    "TRADE_AMT": "120",
                    "TOTAL_VALUE": "1100",
                },
                {
                    "PRODUCT_CODE": "03",
                    "TRADE_AMT": "30",
                    "TOTAL_VALUE": "300",
                },
            ]
        }
    )
    szse = json.dumps(
        [
            {
                "metadata": {
                    "tabkey": "tab1",
                    "conditions": [
                        {"name": "txtQueryDate", "defaultValue": "2026-07"}
                    ],
                },
                "data": [{"zbmc": "成交金额（亿元）", "gp": "327,969.34"}],
            }
        ]
    ).encode("utf-8")
    assert parse_sse_legacy_monthly(legacy, "2015-01")[
        "trade_amount_cny_100m"
    ] == pytest.approx(100.0)
    assert parse_sse_current_monthly(current, "2026-07")[
        "market_cap_cny_100m"
    ] == pytest.approx(1400.0)
    assert parse_szse_monthly(szse, "2026-07")[
        "trade_amount_cny_100m"
    ] == pytest.approx(327969.34)


def test_sse_legacy_monthly_collapses_only_identical_duplicate_rows() -> None:
    row = {
        "PRODUCT_TYPE": "1",
        "TX_AMOUNT": "51,501.9642",
        "MKT_VALUE": "329,219.10",
    }
    payload = _jsonp({"result": [row, dict(row), dict(row), dict(row)]})
    parsed = parse_sse_legacy_monthly(payload, "2017-11")
    assert parsed["trade_amount_cny_100m"] == pytest.approx(51501.9642)
    assert parsed["market_cap_cny_100m"] == pytest.approx(329219.10)
    assert parsed["identical_duplicate_rows_collapsed"] == 3


def test_sse_legacy_monthly_rejects_nonidentical_duplicate_rows() -> None:
    payload = _jsonp(
        {
            "result": [
                {
                    "PRODUCT_TYPE": "1",
                    "TX_AMOUNT": "100",
                    "MKT_VALUE": "1000",
                },
                {
                    "PRODUCT_TYPE": "1",
                    "TX_AMOUNT": "101",
                    "MKT_VALUE": "1000",
                },
            ]
        }
    )
    with pytest.raises(ContractError, match="数值不一致"):
        parse_sse_legacy_monthly(payload, "2017-11")


def test_monthly_factor_uses_external_formula_without_return_fit() -> None:
    config = load_config()
    dates = pd.to_datetime(["2014-12-30", "2014-12-31", "2015-01-05"])
    daily = pd.DataFrame(
        {
            "date": dates,
            "sse_a_trade_amount_cny_100m": [50.0, 50.0, 60.0],
            "sse_a_market_cap_cny_100m": [400.0, 400.0, 410.0],
            "szse_stock_trade_amount_cny_100m": [25.0, 25.0, 30.0],
            "szse_stock_market_cap_cny_100m": [600.0, 600.0, 590.0],
        }
    )
    monthly = build_monthly_factor(daily, config)
    december = monthly.loc[monthly["factor_month"] == "2014-12"].iloc[0]
    assert december["aggregate_turnover_ratio"] == pytest.approx(0.15)
    expected_forecast = -0.019 + 0.258 * 0.15
    assert december["external_forecast_monthly_return"] == pytest.approx(
        expected_forecast
    )
    expected_weight = expected_forecast / (5.0 * 0.0901**2)
    assert december["target_weight_before_lot_rounding"] == pytest.approx(
        expected_weight
    )


def test_monthly_component_factor_rebuilds_aggregates_and_external_rule() -> None:
    config = deepcopy(load_config())
    config["data_contract"]["expected_complete_months"] = 2
    config["data_contract"]["expected_first_factor_month"] = "2014-12"
    config["data_contract"]["expected_last_factor_month"] = "2015-01"
    components = pd.DataFrame(
        {
            "factor_month": ["2014-12", "2015-01"],
            "month_end_date": pd.to_datetime(["2014-12-31", "2015-01-30"]),
            "sse_a_monthly_trade_amount_cny_100m": [100.0, 120.0],
            "sse_a_month_end_market_cap_cny_100m": [400.0, 410.0],
            "szse_stock_monthly_trade_amount_cny_100m": [50.0, 60.0],
            "szse_stock_month_end_market_cap_cny_100m": [600.0, 590.0],
            "aggregate_monthly_trade_amount_cny_100m": [150.0, 180.0],
            "aggregate_month_end_market_cap_cny_100m": [1000.0, 1000.0],
        }
    )
    monthly = build_monthly_factor_from_components(components, config)
    assert monthly["aggregate_turnover_ratio"].tolist() == pytest.approx([0.15, 0.18])
    assert monthly.loc[0, "external_forecast_monthly_return"] == pytest.approx(
        -0.019 + 0.258 * 0.15
    )
    broken = components.copy()
    broken.loc[0, "aggregate_monthly_trade_amount_cny_100m"] = 149.0
    with pytest.raises(ContractError, match="总成交额"):
        build_monthly_factor_from_components(broken, config)


def test_pdf_numeric_row_uses_sequential_decimal_closure() -> None:
    words = [
        {"text": "32", "x0": 10.0, "x1": 14.0},
        {"text": "417", "x0": 15.0, "x1": 21.0},
        {"text": "840.93", "x0": 22.0, "x1": 34.0},
        {"text": "25.070.813.68", "x0": 60.0, "x1": 88.0},
    ]
    values = _parse_sequential_decimal_values(
        words,
        lower_x=0.0,
        upper_x=100.0,
        expected_count=2,
    )
    assert values == pytest.approx([32417840.93, 25070813.68])


def test_static_month_parser_reconciles_stocks_components() -> None:
    listed = b"""<html><body><p>End of July 2026</p><table>
    <tr><td>x</td><td>Stocks</td><td>3</td><td>1,000</td></tr>
    <tr><td>x</td><td>MB A-shares</td><td>1</td><td>600</td></tr>
    <tr><td>x</td><td>MB B-shares</td><td>1</td><td>50</td></tr>
    <tr><td>x</td><td>MB CDR</td><td>0</td><td>0</td></tr>
    <tr><td>x</td><td>ChiNext A-shares</td><td>1</td><td>350</td></tr>
    <tr><td>x</td><td>ChiNext CDR</td><td>0</td><td>0</td></tr>
    </table></body></html>"""
    transaction = b"""<html><body><p>July 2026</p><table>
    <tr><td>x</td><td>Stocks</td><td>23</td><td>500</td></tr>
    <tr><td>x</td><td>MB A-shares</td><td>23</td><td>300</td></tr>
    <tr><td>x</td><td>MB B-shares</td><td>23</td><td>10</td></tr>
    <tr><td>x</td><td>MB CDR</td><td>23</td><td>0</td></tr>
    <tr><td>x</td><td>ChiNext A-shares</td><td>23</td><td>190</td></tr>
    <tr><td>x</td><td>ChiNext CDR</td><td>23</td><td>0</td></tr>
    </table></body></html>"""
    row, detail = parse_szse_static_month(
        listed,
        transaction,
        month="2026-07",
    )
    assert row["szse_stock_month_end_market_cap_cny_100m"] == pytest.approx(1e-5)
    assert row["szse_stock_monthly_trade_amount_cny_100m"] == pytest.approx(5e-6)
    assert detail["listed_component_difference_cny_yuan"] == 0
    assert detail["transaction_component_difference_cny_yuan"] == 0


def test_external_rule_at_paper_mean_turnover() -> None:
    forecast = -0.019 + 0.258 * 0.12
    weight = forecast / (5.0 * 0.0901**2)
    assert forecast == pytest.approx(0.01196)
    assert weight == pytest.approx(0.2946534927)


def test_signal_executes_on_next_real_trading_open() -> None:
    config = load_config()
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2014-12-30", "2014-12-31", "2015-01-05", "2015-01-06"]
            )
        }
    )
    monthly = pd.DataFrame(
        {
            "factor_month": ["2014-12"],
            "month_end_date": pd.to_datetime(["2014-12-31"]),
            "aggregate_turnover_ratio": [0.12],
            "external_forecast_monthly_return": [0.01196],
            "target_weight_before_lot_rounding": [0.2946521988],
        }
    )
    signals = build_signal_schedule(market, monthly, config)
    assert signals.loc[0, "decision_date"] == pd.Timestamp("2014-12-31")
    assert signals.loc[0, "execution_date"] == pd.Timestamp("2015-01-05")


def test_mechanism_target_uses_open_to_next_open_and_record_date_entitlement() -> None:
    config = load_config()
    signals = pd.DataFrame(
        {
            "factor_month": ["2014-12", "2015-01", "2015-02"],
            "decision_date": pd.to_datetime(
                ["2014-12-31", "2015-01-30", "2015-02-27"]
            ),
            "execution_date": pd.to_datetime(
                ["2015-01-05", "2015-02-02", "2015-03-02"]
            ),
            "aggregate_turnover_ratio": [0.12, 0.13, 0.14],
            "external_forecast_monthly_return": [0.01, 0.02, 0.03],
            "target_position": [0.25, 0.50, 0.75],
            "signal_reason": ["测试"] * 3,
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-05", "2015-02-02", "2015-03-02"]),
            "open": [4.0, 4.2, 4.1],
        }
    )
    dividends = pd.DataFrame(
        {
            "record_date": pd.to_datetime(["2015-01-15", "2015-02-02"]),
            "cash_dividend_per_share": [0.10, 0.20],
        }
    )
    mechanism = add_mechanism_targets(signals, market, dividends, config)
    assert len(mechanism) == 2
    assert mechanism.loc[0, "entitled_cash_dividend_per_share"] == pytest.approx(0.10)
    assert mechanism.loc[0, "future_interval_total_return"] == pytest.approx(
        (4.2 + 0.10) / 4.0 - 1.0
    )
    assert mechanism.loc[1, "entitled_cash_dividend_per_share"] == pytest.approx(0.20)


def test_newey_west_and_block_bootstrap_detect_positive_slope() -> None:
    rng = np.random.default_rng(20260830)
    x = np.linspace(0.05, 0.40, 180)
    y = 0.30 * x + rng.normal(0.0, 0.005, len(x))
    regression = newey_west_slope(pd.Series(x), pd.Series(y), max_lag=3)
    assert regression["slope"] is not None
    assert regression["slope"] > 0.0
    assert regression["t_stat"] is not None
    assert regression["t_stat"] > 1.645
    config = load_config()
    reduced = deepcopy(config)
    reduced["bootstrap"]["repetitions"] = 300
    frame = pd.DataFrame(
        {
            "aggregate_turnover_ratio": x,
            "future_interval_total_return": y,
        }
    )
    bootstrap = circular_block_bootstrap_slope(frame, reduced)
    assert bootstrap["valid_draws"] == 300
    assert bootstrap["confidence_interval"][0] > 0.0
