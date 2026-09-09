"""经典技术方向规则冻结测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.option_technical_rule_discovery_v1 import build_rule_sides


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_technical_rule_discovery_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_exactly_four_frozen_rules_and_at_most_ten_factors(config: dict) -> None:
    assert sorted(config["candidates"]) == [
        "R1_TSMOM_120",
        "R2_MA_20_120",
        "R3_DONCHIAN_55_20",
        "R4_MACD_12_26_9",
    ]
    assert config["factor_budget"]["distinct_used"] == 5
    assert config["factor_budget"]["distinct_used"] <= 10


def test_uptrend_rules_all_choose_calls_after_warmup() -> None:
    dates = pd.bdate_range("2020-01-01", periods=180)
    benchmark = pd.DataFrame(
        {"date": dates, "close": 4000.0 * np.exp(np.arange(len(dates)) * 0.001)}
    )
    rules = build_rule_sides(benchmark)
    last = rules.iloc[-1]
    for rule_id in (
        "R1_TSMOM_120",
        "R2_MA_20_120",
        "R3_DONCHIAN_55_20",
        "R4_MACD_12_26_9",
    ):
        assert last[rule_id] == "C"


def test_downtrend_rules_all_choose_puts_after_warmup() -> None:
    dates = pd.bdate_range("2020-01-01", periods=180)
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": 4000.0
            * np.exp(-0.0001 * np.power(np.arange(len(dates), dtype=float), 1.35)),
        }
    )
    rules = build_rule_sides(benchmark)
    last = rules.iloc[-1]
    for rule_id in (
        "R1_TSMOM_120",
        "R2_MA_20_120",
        "R3_DONCHIAN_55_20",
        "R4_MACD_12_26_9",
    ):
        assert last[rule_id] == "P"
