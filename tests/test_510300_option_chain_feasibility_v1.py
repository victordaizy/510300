from __future__ import annotations

import pandas as pd
import pytest

from research.option_chain_feasibility_v1 import (
    assert_outcome_blind_columns,
    audit_adjustment_ledger,
    audit_maturity_surface,
    build_surface_audits,
    decide_terminal_status,
)


def geometry() -> dict:
    return {
        "price_field": "settlement",
        "adjusted_contracts_allowed_in_surface_geometry": False,
        "minimum_days_to_expiry": 7,
        "maximum_days_to_expiry": 190,
        "near_30d_minimum_dte": 20,
        "near_30d_maximum_dte": 45,
        "near_60d_minimum_dte": 46,
        "near_60d_maximum_dte": 90,
        "minimum_valid_maturities_per_day": 2,
        "minimum_matched_call_put_strikes_per_maturity": 3,
        "minimum_otm_puts_per_maturity": 2,
        "minimum_otm_calls_per_maturity": 2,
        "maximum_atm_moneyness_gap": 0.05,
        "target_put_absolute_delta": 0.25,
        "minimum_positive_iv_ratio": 0.90,
        "maximum_put_call_parity_forward_dispersion_ratio": 0.03,
        "monotonicity_price_tolerance": 0.002,
        "convexity_slope_tolerance": 0.05,
        "maximum_obvious_static_arbitrage_violation_ratio": 0.05,
        "price_upper_bound_tolerance_ratio": 0.02,
    }


def synthetic_maturity(trade_date: str, expiry_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    call_prices = [20.5, 10.8, 6.5, 3.0, 1.5, 0.8, 0.3]
    call_deltas = [0.95, 0.80, 0.65, 0.50, 0.35, 0.20, 0.05]
    eod_rows = []
    risk_rows = []
    for option_type in ("C", "P"):
        for index, strike in enumerate(strikes):
            code = f"{trade_date.replace('-', '')}{expiry_date.replace('-', '')}{option_type}{index}.SH"
            call_price = call_prices[index]
            settlement = call_price if option_type == "C" else call_price - (100.0 - strike)
            delta = call_deltas[index] if option_type == "C" else call_deltas[index] - 1.0
            eod_rows.append(
                {
                    "trade_date": trade_date,
                    "contract_code": code,
                    "option_type": option_type,
                    "expiry_date": expiry_date,
                    "strike": strike,
                    "contract_unit": 10000,
                    "is_adjusted": False,
                    "settlement": settlement,
                    "underlying_close": 100.0,
                }
            )
            risk_rows.append(
                {
                    "trade_date": trade_date,
                    "contract_code": code,
                    "delta": delta,
                    "theta": -0.01,
                    "gamma": 0.01,
                    "vega": 0.10,
                    "rho": 0.01,
                    "implied_volatility": 0.20,
                }
            )
    return pd.DataFrame(eod_rows), pd.DataFrame(risk_rows)


def test_outcome_blindness_rejects_future_return_column() -> None:
    with pytest.raises(ValueError, match="结果盲输入检查失败"):
        assert_outcome_blind_columns(
            {"bad": pd.DataFrame({"future_return_10d": [0.1]})},
            ["future_return", "sharpe"],
        )


def test_outcome_blindness_accepts_same_day_option_columns() -> None:
    result = assert_outcome_blind_columns(
        {
            "option": pd.DataFrame(
                {"trade_date": ["2026-08-14"], "settlement": [0.1], "delta": [-0.25]}
            )
        },
        ["future_return", "sharpe"],
    )
    assert result["future_data_reads"] == 0
    assert result["violations"] == []


def test_synthetic_maturity_passes_geometry_gates() -> None:
    eod, risk = synthetic_maturity("2026-06-01", "2026-07-01")
    merged = eod.merge(risk, on=["trade_date", "contract_code"], validate="one_to_one")
    merged["days_to_expiry"] = 30
    result = audit_maturity_surface(merged, geometry())
    assert result["valid_maturity"] is True
    assert result["put_25delta_bracket_available"] is True
    assert result["parity_forward_dispersion_ratio"] == pytest.approx(0.0)
    assert result["obvious_static_arbitrage_violation_count"] == 0


def test_surface_day_requires_valid_30d_and_60d_maturities() -> None:
    eod_30, risk_30 = synthetic_maturity("2026-06-01", "2026-07-01")
    eod_60, risk_60 = synthetic_maturity("2026-06-01", "2026-07-31")
    eod = pd.concat([eod_30, eod_60], ignore_index=True)
    risk = pd.concat([risk_30, risk_60], ignore_index=True)
    calendar = pd.DataFrame({"trade_date": ["2026-06-01"]})
    maturities, daily, summary = build_surface_audits(eod, risk, calendar, geometry())
    assert len(maturities) == 2
    assert bool(daily.loc[0, "valid_surface_day"]) is True
    assert summary["valid_surface_day_ratio"] == pytest.approx(1.0)


def test_surface_day_fails_without_60d_bucket() -> None:
    eod, risk = synthetic_maturity("2026-06-01", "2026-07-01")
    calendar = pd.DataFrame({"trade_date": ["2026-06-01"]})
    _, daily, summary = build_surface_audits(eod, risk, calendar, geometry())
    assert bool(daily.loc[0, "valid_surface_day"]) is False
    assert summary["valid_surface_day_ratio"] == pytest.approx(0.0)


def test_missing_adjustment_ledger_is_not_complete(tmp_path) -> None:
    protocol = {
        "inputs": {
            "contract_adjustment_ledger": "missing.parquet",
            "contract_adjustment_evidence": "missing.json",
        }
    }
    master = pd.DataFrame(
        {"contract_code": ["10000001.SH"], "is_adjusted": [True]}
    )
    result = audit_adjustment_ledger(tmp_path, protocol, master)
    assert result["complete"] is False
    assert result["coverage_ratio"] == pytest.approx(0.0)


def test_terminal_status_requires_every_hard_gate() -> None:
    states = {"pass": "PASS", "blocked": "BLOCKED"}
    assert decide_terminal_status({"a": True, "b": True}, states) == "PASS"
    assert decide_terminal_status({"a": True, "b": False}, states) == "BLOCKED"
