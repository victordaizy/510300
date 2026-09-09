"""GammaOIPressure迁移协议的安全门与执行测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.option_gamma_oi_pressure_v1 import (
    build_schedule,
    compute_gamma_oi_pressure,
    execute_trade,
    select_contract,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_gamma_oi_pressure_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def factor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": [0.20, 0.30],
            "low": [0.10, 0.20],
            "turnover_10k_cny": [100.0, 200.0],
            "open_interest": [1200.0, 900.0],
            "lag1_open_interest": [1000.0, 1000.0],
            "volume": [500.0, 500.0],
            "delta": [0.5, -0.5],
            "underlying_close": [4.0, 4.0],
            "gamma": [1.2, 1.2],
            "theta": [-0.2, -0.2],
            "dte_years": [30 / 365.2425, 30 / 365.2425],
        }
    )


def test_factor_matches_frozen_formula(config: dict) -> None:
    frame = factor_frame()
    actual = compute_gamma_oi_pressure(frame, config["factor"]["epsilon"])
    expected_first = (
        -((0.20 - 0.10) / (100.0 * 10000.0 + 1.0e-12))
        * np.tanh((1200.0 - 1000.0) / 501.0)
        * np.tanh(0.5)
        * np.log1p(4.0**2 * 1.2)
        / np.sqrt(1.0 + 0.2)
        / np.sqrt(1.0 + 30 / 365.2425)
    )
    assert actual.iloc[0] == pytest.approx(expected_first)
    assert np.isfinite(actual).all()


def test_factor_has_no_future_row_dependency(config: dict) -> None:
    original = factor_frame()
    before = compute_gamma_oi_pressure(original, config["factor"]["epsilon"])
    extended = pd.concat(
        [original, original.iloc[[0]].assign(open_interest=999999.0)], ignore_index=True
    )
    after = compute_gamma_oi_pressure(extended, config["factor"]["epsilon"])
    pd.testing.assert_series_equal(before, after.iloc[: len(before)], check_names=False)


def test_selection_uses_highest_factor_and_deterministic_tie(config: dict) -> None:
    rows = pd.DataFrame(
        {
            "contract_code": ["1002.SH", "1001.SH", "1003.SH"],
            "option_type": ["C", "P", "C"],
            "is_adjusted": [False, False, False],
            "contract_unit": [10000, 10000, 10000],
            "dte": [30, 30, 30],
            "moneyness": [1.0, 1.0, 1.0],
            "volume": [500, 500, 500],
            "open_interest": [1000, 1000, 1000],
            "signal_premium_per_contract": [3000.0, 3000.0, 3000.0],
            "gamma_oi_pressure": [0.5, 0.5, 0.4],
            "close": [0.3, 0.3, 0.3],
            "delta": [0.5, -0.5, 0.4],
            "gamma": [1.0, 1.0, 1.0],
            "theta": [-0.2, -0.2, -0.2],
        }
    )
    selected = select_contract(rows, config)
    assert selected is not None
    assert selected["contract_code"] == "1001.SH"
    assert selected["quantity"] == 3


def test_execution_uses_high_low_fees_and_rejects_small_trade(config: dict) -> None:
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "contract_code": ["1001.SH", "1001.SH"],
            "high": [0.40, 0.50],
            "low": [0.35, 0.20],
            "volume": [100, 100],
        }
    ).set_index(["trade_date", "contract_code"])
    selection = {"contract_code": "1001.SH", "quantity": 2, "contract_unit": 10000}
    result = execute_trade(
        selection,
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-01-04"),
        panel,
        20000.0,
        config,
    )
    assert result.executed
    assert result.entry_notional_cny == pytest.approx(8000.0)
    assert result.exit_notional_cny == pytest.approx(4000.0)
    assert result.fees_cny == pytest.approx(20.0)
    assert result.pnl_cny == pytest.approx(-4020.0)
    small = execute_trade(
        {"contract_code": "1001.SH", "quantity": 1, "contract_unit": 10000},
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-01-04"),
        panel,
        20000.0,
        config,
    )
    assert small.status == "SMALL_OPENING_TRADE_REJECTED"
    assert not small.executed


def test_schedule_is_non_overlapping_and_never_crosses_cutoff(config: dict) -> None:
    dates = pd.bdate_range("2023-11-01", "2024-01-31")
    benchmark = pd.DataFrame({"date": dates, "close": np.arange(len(dates)) + 100})
    schedule = build_schedule(benchmark, config)
    assert schedule
    assert max(item["exit_date"] for item in schedule) <= pd.Timestamp("2023-12-29")
    assert all(item["entry_date"] > item["signal_date"] for item in schedule)
    assert all(item["exit_date"] > item["entry_date"] for item in schedule)
    assert all(
        schedule[index + 1]["entry_date"] > schedule[index]["exit_date"]
        for index in range(len(schedule) - 1)
    )
