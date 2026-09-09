"""510300流动性压力风险开关长周期复制的必要契约测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.liquidity_stress_long_cycle_replication_v1 import (
    cost_model,
    load_and_validate_config,
    load_inputs,
)


def test_frozen_account_contract_and_cost_scenarios() -> None:
    """新结果必须严格使用20万元万二合同及预声明压力成本。"""

    config, hashes = load_and_validate_config()
    assert config["protocol"]["project_id"] == (
        "510300_LIQUIDITY_STRESS_LONG_CYCLE_REPLICATION_V1"
    )
    assert hashes
    assert config["execution"]["initial_capital_cny"] == pytest.approx(200000.0)
    base = cost_model(config, "BASE")
    stress = cost_model(config, "STRESS")
    assert base.commission_rate == pytest.approx(0.0002)
    assert base.minimum_commission == pytest.approx(5.0)
    assert base.slippage_bps == pytest.approx(5.0)
    assert stress.commission_rate == pytest.approx(0.0004)
    assert stress.minimum_commission == pytest.approx(5.0)
    assert stress.slippage_bps == pytest.approx(10.0)


def test_same_definition_history_is_continuous_without_fill() -> None:
    """2015评价前必须有连续暖机历史，拼接日同字段一致且不得填充。"""

    config, _ = load_and_validate_config()
    signal, market, dividends, audit = load_inputs(config)
    assert audit["status"] == "PASS_SAME_DEFINITION_CONTINUOUS_HISTORY"
    assert audit["signal_first_date"] == "2012-07-02"
    assert audit["signal_last_date"] == "2026-08-12"
    assert audit["market_last_date"] == "2026-08-13"
    assert audit["missing_signal_rows"] == 0
    assert audit["static_fill_rows"] == 0
    assert audit["proxy_substitution_rows"] == 0
    assert audit["margin_seam_absolute_differences"]["rzye"] == pytest.approx(0.0)
    assert audit["margin_seam_absolute_differences"]["rqyl"] == pytest.approx(0.0)
    assert audit["margin_seam_absolute_differences"]["rqye"] <= 0.0001
    assert signal["date"].is_monotonic_increasing
    assert market["date"].is_monotonic_increasing
    assert not signal.isna().any(axis=None)
    assert not market.isna().any(axis=None)
    assert not dividends.empty
