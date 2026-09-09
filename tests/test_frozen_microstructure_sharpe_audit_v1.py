from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import frozen_microstructure_sharpe_audit_v1 as audit


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )


def test_config_freezes_only_510300_and_post_2021_scope() -> None:
    config = audit.load_config()
    assert config["scope"]["execution_asset"] == "510300.SH"
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["scope"]["earliest_allowed_source_date"] == "2021-01-01"
    assert config["execution"]["initial_capital_cny"] == 20000.0
    assert config["execution"]["cash_annual_rate"] == 0.0


def test_commission_applies_minimum_five_yuan() -> None:
    costs = {
        "commission_rate_per_leg": 0.0003,
        "minimum_commission_cny_per_leg": 5.0,
        "slippage_bps_per_leg": 5.0,
    }
    assert audit._commission(1000.0, costs) == 5.0
    assert audit._commission(100000.0, costs) == pytest.approx(30.0)


def test_simulation_respects_lot_size_costs_and_t_plus_one_state() -> None:
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-05", "2022-01-06"]),
            "etf_open": [4.0, 4.0, 4.0],
            "etf_close": [4.0, 4.1, 4.0],
            "benchmark_close": [100.0, 101.0, 100.0],
        }
    )
    costs = {
        "commission_rate_per_leg": 0.0003,
        "minimum_commission_cny_per_leg": 5.0,
        "slippage_bps_per_leg": 5.0,
    }
    ledger, trades = audit.simulate(
        sample,
        _empty_dividends(),
        np.array([0, 1, 0], dtype=np.int8),
        costs=costs,
        initial_capital=20000.0,
        lot_size=100,
    )
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert (trades["quantity"] % 100 == 0).all()
    assert trades["commission"].tolist() == pytest.approx(
        (trades["execution_notional"] * 0.0003).tolist()
    )
    assert (trades["commission"] > 5.0).all()
    assert ledger["policy_state"].tolist() == [0, 1, 0]


def test_round_trip_is_counted_when_cash_state_recovers() -> None:
    states = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-01-04", "2022-01-05", "2022-01-06", "2023-01-03"]
            ),
            "policy_state": [1, 0, 0, 1],
        }
    )
    assert audit._complete_round_trips_by_year(states) == {"2023": 1}


def test_manifest_rejects_hash_drift_when_present() -> None:
    if not audit.MANIFEST_PATH.exists():
        return
    config = audit.load_config()
    manifest = json.loads(audit.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == audit.sha256_file(audit.CONFIG_PATH)
    validated = audit.validate_manifest(config)
    assert validated["implementation_frozen"] is True
