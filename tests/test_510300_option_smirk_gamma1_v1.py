"""风险中性偏度Gamma1迁移协议测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.option_smirk_gamma1_v1 import (
    build_schedule,
    execute_trade,
    interpolate_constant_maturity,
    weighted_smirk_fit,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_smirk_gamma1_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_weighted_smirk_recovers_known_gamma_coefficients() -> None:
    xi = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    gamma0, gamma1, gamma2 = 0.20, -0.30, 0.40
    iv = gamma0 * (1.0 + gamma1 * xi + gamma2 * xi**2)
    fitted = weighted_smirk_fit(xi, iv, np.array([1, 2, 3, 4, 5], dtype=float))
    assert fitted is not None
    assert fitted[0] == pytest.approx(gamma0)
    assert fitted[1] == pytest.approx(gamma1)
    assert fitted[2] == pytest.approx(gamma2)


def test_constant_maturity_interpolation_uses_bracketing_expiries() -> None:
    rows = pd.DataFrame({"dte": [20.0, 40.0, 70.0], "gamma1": [-0.4, 0.2, 0.5]})
    assert interpolate_constant_maturity(rows, 30) == pytest.approx(-0.1)
    assert interpolate_constant_maturity(rows, 10) is None


def test_execution_is_pessimistic_and_filters_small_orders(config: dict) -> None:
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2023-01-03", "2023-02-01"]),
            "contract_code": ["1001.SH", "1001.SH"],
            "high": [0.40, 0.50],
            "low": [0.35, 0.20],
            "volume": [100, 100],
        }
    ).set_index(["trade_date", "contract_code"])
    result = execute_trade(
        {"contract_code": "1001.SH", "quantity": 2, "contract_unit": 10000},
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-02-01"),
        panel,
        20000.0,
        config,
    )
    assert result.executed
    assert result.entry_notional_cny == pytest.approx(8000.0)
    assert result.exit_notional_cny == pytest.approx(4000.0)
    assert result.pnl_cny == pytest.approx(-4020.0)
    small = execute_trade(
        {"contract_code": "1001.SH", "quantity": 1, "contract_unit": 10000},
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-02-01"),
        panel,
        20000.0,
        config,
    )
    assert small.status == "SMALL_OPENING_TRADE_REJECTED"
    assert not small.executed


def test_schedule_is_non_overlapping_and_stays_in_development(config: dict) -> None:
    dates = pd.bdate_range("2020-01-01", "2024-01-31")
    benchmark = pd.DataFrame({"date": dates, "close": np.arange(len(dates)) + 100})
    schedule = build_schedule(benchmark, config)
    assert schedule
    assert max(item["exit_date"] for item in schedule) <= pd.Timestamp("2023-12-29")
    assert all(item["entry_date"] > item["signal_date"] for item in schedule)
    assert all(
        schedule[index + 1]["entry_date"] > schedule[index]["exit_date"]
        for index in range(len(schedule) - 1)
    )
