"""交易所聚合IVS二元筛选的无前视与状态机测试。"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from research.exchange_ivs_etf_binary_screen_v1 import (
    build_candidate_states,
    build_exchange_ivs_feature,
    expanding_ols_forecasts,
    load_config,
)


def test_expanding_ols_only_uses_mature_labels() -> None:
    dates = pd.date_range("2020-01-03", periods=12, freq="7D")
    endpoints = pd.DataFrame(
        {
            "trade_date": dates,
            "exchange_ivs": np.linspace(-0.10, 0.05, len(dates)),
            "benchmark_close": 100.0 * np.exp(np.linspace(0.0, 0.11, len(dates))),
            "market_position": np.arange(len(dates), dtype=int) * 5,
        }
    )
    forecasts = expanding_ols_forecasts(
        endpoints,
        candidate_id="TEST_2W",
        schedule="WEEKLY_LAST_TRADING_DAY",
        horizon_periods=2,
        minimum_training_observations=3,
        cash_annual_rate=0.015,
        trading_days_per_year=242,
    )
    assert forecasts.loc[3, "status"] == "NO_VIEW_INSUFFICIENT_MATURE_LABELS"
    assert forecasts.loc[4, "status"] == "MODEL_READY"
    assert int(forecasts.loc[4, "training_observations"]) == 3
    assert forecasts.loc[4, "maximum_training_label_end_date"] == dates[4]
    ready = forecasts.loc[forecasts["status"].eq("MODEL_READY")]
    assert (ready["maximum_training_label_end_date"] <= ready["signal_date"]).all()


def test_ready_close_state_only_affects_following_market_rows_through_ledger() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "open": 4.0,
            "close": 4.0,
        }
    )
    forecasts = pd.DataFrame(
        [
            {
                "candidate_id": "TEST",
                "signal_date": market.loc[1, "trade_date"],
                "status": "MODEL_READY",
                "target_state": 1,
            },
            {
                "candidate_id": "TEST",
                "signal_date": market.loc[3, "trade_date"],
                "status": "MODEL_READY",
                "target_state": 0,
            },
        ]
    )
    states = build_candidate_states(market, forecasts, ["TEST"])["TEST"]
    assert states.tolist() == [0, 1, 1, 0, 0]
    assert states[0] == 0
    assert states[1] == 1


def test_exchange_ivs_feature_exports_no_basis_signal() -> None:
    config = copy.deepcopy(load_config())
    config["ivs_construction"]["required_feature_days"] = 1
    trade_date = pd.Timestamp("2024-01-02")
    expiry_date = pd.Timestamp("2024-02-01")
    dte = (expiry_date - trade_date).days
    spot = 4.0
    strike = 4.0
    rate = 0.02
    put_close = 0.10
    call_close = spot - strike * math.exp(-rate * dte / 365.25) + put_close
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
    feature, audit = build_exchange_ivs_feature(option_eod, risk, curve, config)
    assert math.isclose(float(feature.loc[0, "exchange_ivs"]), 0.01)
    assert "pair_basis" not in feature.columns
    assert "obasis" not in feature.columns
    assert audit["basis_signal_columns_exported"] == 0
