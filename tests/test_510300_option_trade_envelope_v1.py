"""510300期权不利成交价包络开发协议测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.option_trade_envelope_v1 import (
    build_schedule,
    evaluate_leg,
    select_contract,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_trade_envelope_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_contract_selection_is_signal_day_only_and_deterministic(config: dict) -> None:
    rows = pd.DataFrame(
        {
            "contract_code": ["1001.SH", "1002.SH", "1003.SH"],
            "option_type": ["C", "C", "P"],
            "is_adjusted": [False, False, False],
            "contract_unit": [10000, 10000, 10000],
            "dte": [45, 45, 45],
            "delta": [0.84, 0.86, -0.85],
            "volume": [1000, 900, 1000],
            "open_interest": [2000, 2000, 2000],
            "close": [0.30, 0.30, 0.30],
            "premium_per_contract": [3000.0, 3000.0, 3000.0],
        }
    )
    selected = select_contract(rows, "C", config)
    assert selected is not None
    assert selected["contract_code"] == "1001.SH"
    assert selected["quantity"] == 2
    assert selected["signal_premium"] == pytest.approx(6000.0)


def test_execution_uses_entry_high_exit_low_and_both_leg_fees(config: dict) -> None:
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2023-01-03", "2023-01-17"]),
            "contract_code": ["1001.SH", "1001.SH"],
            "high": [0.40, 0.50],
            "low": [0.35, 0.20],
            "volume": [100, 100],
        }
    ).set_index(["trade_date", "contract_code"])
    selection = {"contract_code": "1001.SH", "quantity": 2, "contract_unit": 10000}
    result = evaluate_leg(
        selection,
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-01-17"),
        panel,
        20000.0,
        config,
    )
    assert result.executed
    assert result.entry_notional_cny == pytest.approx(8000.0)
    assert result.exit_notional_cny == pytest.approx(4000.0)
    assert result.pnl_cny == pytest.approx(-4020.0)
    assert result.entry_minimum_pass


def test_missing_exit_is_full_premium_loss_plus_fees(config: dict) -> None:
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2023-01-03"]),
            "contract_code": ["1001.SH"],
            "high": [0.40],
            "low": [0.35],
            "volume": [100],
        }
    ).set_index(["trade_date", "contract_code"])
    selection = {"contract_code": "1001.SH", "quantity": 1, "contract_unit": 10000}
    result = evaluate_leg(
        selection,
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-01-17"),
        panel,
        20000.0,
        config,
    )
    assert result.status == "MISSING_EXIT_MARKED_ZERO"
    assert result.pnl_cny == pytest.approx(-4010.0)


def test_schedule_never_crosses_development_end(config: dict) -> None:
    dates = pd.bdate_range("2023-11-01", "2024-01-31")
    benchmark = pd.DataFrame({"date": dates, "close": range(len(dates))})
    schedule = build_schedule(benchmark, config)
    assert schedule
    assert max(row["exit_date"] for row in schedule) <= pd.Timestamp("2023-12-29")
    assert all(row["entry_date"] > row["signal_date"] for row in schedule)
