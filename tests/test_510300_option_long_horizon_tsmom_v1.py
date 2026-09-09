"""长周期深度实值期权时间序列动量测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.option_long_horizon_tsmom_v1 import build_schedule, momentum_sides
from research.option_trade_envelope_v1 import select_contract


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_long_horizon_tsmom_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_one_factor_and_fixed_40_day_horizon(config: dict) -> None:
    assert config["protocol"]["factor_count"] == 1
    assert config["protocol"]["factor_count"] <= 10
    assert config["schedule"]["holding_trading_days"] == 40
    assert config["factor"]["parameter_search_allowed"] is False


def test_schedule_is_nonoverlapping_and_stays_in_development(config: dict) -> None:
    dates = pd.bdate_range("2020-01-01", "2023-01-31")
    benchmark = pd.DataFrame({"date": dates, "close": range(1, len(dates) + 1)})
    schedule = build_schedule(benchmark, config)
    assert schedule
    assert max(row["exit_date"] for row in schedule) <= pd.Timestamp("2022-12-30")
    for left, right in zip(schedule, schedule[1:]):
        assert right["entry_date"] > left["exit_date"]


def test_120_day_momentum_direction() -> None:
    dates = pd.bdate_range("2020-01-01", periods=160)
    up = pd.DataFrame({"date": dates, "close": np.arange(1, 161, dtype=float)})
    down = pd.DataFrame({"date": dates, "close": np.arange(160, 0, -1, dtype=float)})
    assert momentum_sides(up).iloc[-1] == "C"
    assert momentum_sides(down).iloc[-1] == "P"


def test_long_dated_contract_selection_is_deterministic(config: dict) -> None:
    rows = pd.DataFrame(
        {
            "contract_code": ["1001.SH", "1002.SH"],
            "option_type": ["C", "C"],
            "is_adjusted": [False, False],
            "contract_unit": [10000, 10000],
            "dte": [100, 100],
            "delta": [0.84, 0.86],
            "volume": [1000, 900],
            "open_interest": [2000, 2000],
            "close": [0.35, 0.35],
            "premium_per_contract": [3500.0, 3500.0],
        }
    )
    selected = select_contract(rows, "C", config)
    assert selected is not None
    assert selected["contract_code"] == "1001.SH"
    assert selected["quantity"] == 2
