"""期权—现货基差特征构造的确定性单元测试。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.option_cash_basis_feature_audit_v1 import (
    aggregate_daily_features,
    construct_pair_basis,
    interpolate_maturity_rate,
    normalize_date,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "510300_option_cash_basis_feature_audit_v1.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def _synthetic_inputs(call_price_increment: float = 0.0):
    trade_date = pd.Timestamp("2024-01-02")
    expiry_date = pd.Timestamp("2024-02-01")
    dte = (expiry_date - trade_date).days
    spot = 4.0
    strike = 4.0
    rate = 0.02
    put_close = 0.10
    call_close = spot - strike * math.exp(-rate * dte / 365.25) + put_close
    call_close += call_price_increment
    option_eod = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "contract_code": "CALL",
                "option_type": "C",
                "expiry_date": expiry_date,
                "strike": strike,
                "contract_unit": 10000,
                "is_adjusted": False,
                "close": call_close,
                "volume": 100,
                "open_interest": 500,
                "underlying_close": spot,
            },
            {
                "trade_date": trade_date,
                "contract_code": "PUT",
                "option_type": "P",
                "expiry_date": expiry_date,
                "strike": strike,
                "contract_unit": 10000,
                "is_adjusted": False,
                "close": put_close,
                "volume": 90,
                "open_interest": 400,
                "underlying_close": spot,
            },
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "contract_code": "CALL",
                "implied_volatility": 0.25,
            },
            {
                "trade_date": trade_date,
                "contract_code": "PUT",
                "implied_volatility": 0.24,
            },
        ]
    )
    curve = pd.DataFrame(
        [
            {
                "date": trade_date,
                "cgb_3m": 2.0,
                "cgb_6m": 2.0,
                "cgb_1y": 2.0,
            }
        ]
    )
    return option_eod, risk, curve


def test_maturity_rate_interpolation_and_edge_clamping() -> None:
    result = interpolate_maturity_rate(
        np.array([30.0, 91.3125, 136.96875, 365.25, 500.0]),
        np.full(5, 1.0),
        np.full(5, 2.0),
        np.full(5, 4.0),
    )
    np.testing.assert_allclose(result, [0.01, 0.01, 0.015, 0.04, 0.04])


def test_date_normalization_unifies_source_precisions() -> None:
    microseconds = pd.Series(np.array(["2024-01-02"], dtype="datetime64[us]"))
    milliseconds = pd.Series(np.array(["2024-01-02"], dtype="datetime64[ms]"))
    assert normalize_date(microseconds).dtype == "datetime64[ns]"
    assert normalize_date(milliseconds).dtype == "datetime64[ns]"


def test_put_call_parity_produces_zero_basis() -> None:
    option_eod, risk, curve = _synthetic_inputs()
    pairs, audit = construct_pair_basis(option_eod, risk, curve, _config())
    assert audit["final_valid_pairs"] == 1
    assert abs(float(pairs.loc[0, "pair_basis"])) < 1e-12
    daily = aggregate_daily_features(pairs)
    assert abs(float(daily.loc[0, "obasis"])) < 1e-12
    assert math.isclose(float(daily.loc[0, "exchange_ivs"]), 0.01)


def test_call_premium_above_parity_produces_positive_basis() -> None:
    option_eod, risk, curve = _synthetic_inputs(call_price_increment=0.01)
    pairs, audit = construct_pair_basis(option_eod, risk, curve, _config())
    assert audit["final_valid_pairs"] == 1
    assert float(pairs.loc[0, "option_implied_spot"]) > 4.0
    assert float(pairs.loc[0, "pair_basis"]) > 0.0
