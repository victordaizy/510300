import numpy as np
import pandas as pd
import pytest

from research.valuation_v2_expected_return import (
    derive_industry_aware_normalization,
    evaluate_expected_edge,
    expanding_conditioned_fair_pe_v2,
    horizon_return_components,
    normalization_method,
)


def _annual_company_rows(con_code: str, profits: list[float]) -> list[dict]:
    rows = []
    for offset, profit in enumerate(profits):
        year = 2022 + offset
        period = pd.Timestamp(f"{year}-12-31")
        rows.append(
            {
                "con_code": con_code,
                "report_period": period,
                "available_at": period + pd.Timedelta(days=60),
                "revenue_cny": 100.0,
                "net_profit_parent_cny": profit,
                "equity_parent_cny": 100.0 + 10.0 * offset,
                "total_shares": 10.0,
            }
        )
    return rows


def test_industry_method_router_is_fixed() -> None:
    assert normalization_method("银行") == "FINANCIAL_ROE"
    assert normalization_method("有色金属") == "CYCLICAL_MARGIN"
    assert normalization_method("食品饮料") == "OTHER_BLEND"
    assert normalization_method(None) == "OTHER_BLEND"


def test_industry_aware_normalization_reduces_peak_profit() -> None:
    financials = pd.DataFrame(
        _annual_company_rows("BANK", [10.0, 11.0, 12.0, 50.0])
        + _annual_company_rows("METAL", [10.0, 12.0, 15.0, 40.0])
    )
    mapping = pd.DataFrame(
        {
            "con_code": ["BANK", "METAL"],
            "industry_l1": ["银行", "有色金属"],
        }
    )
    result = derive_industry_aware_normalization(financials, mapping)
    bank = result.set_index("con_code").loc["BANK"]
    metal = result.set_index("con_code").loc["METAL"]
    assert bank["normalization_method"] == "FINANCIAL_ROE"
    assert metal["normalization_method"] == "CYCLICAL_MARGIN"
    assert bank["normalized_profit_cny"] < bank["ttm_profit_cny"]
    assert metal["normalized_profit_cny"] < metal["ttm_profit_cny"]
    assert -0.10 <= bank["expected_earnings_growth_annual"] <= 0.15


def test_fair_pe_does_not_use_signal_day_price_or_pe() -> None:
    dates = pd.date_range("2018-01-31", periods=60, freq="ME")
    market = pd.DataFrame(
        {
            "date": dates,
            "index_close": np.linspace(3000.0, 5000.0, len(dates)),
            "pe_ttm": np.linspace(10.0, 14.0, len(dates)),
            "implied_roe": np.linspace(0.10, 0.14, len(dates)),
            "eps_growth_12m": np.linspace(-0.05, 0.10, len(dates)),
            "cgb_10y": np.linspace(3.5, 2.0, len(dates)),
            "cgb_term_spread": np.linspace(0.5, 0.8, len(dates)),
            "realized_volatility_3m": np.linspace(0.25, 0.15, len(dates)),
        }
    )
    fundamentals = pd.DataFrame(
        {
            "date": [dates[-1]],
            "weighted_normalized_roe": [0.13],
            "expected_earnings_growth_annual": [0.06],
        }
    )
    original = expanding_conditioned_fair_pe_v2(
        market, fundamentals, minimum_training_samples=48
    )
    perturbed = market.copy()
    perturbed.loc[perturbed.index[-1], ["index_close", "pe_ttm"]] = [99999.0, 99.0]
    changed = expanding_conditioned_fair_pe_v2(
        perturbed, fundamentals, minimum_training_samples=48
    )
    assert original.loc[0, "conditioned_fair_pe_v2"] == pytest.approx(
        changed.loc[0, "conditioned_fair_pe_v2"]
    )
    assert original.loc[0, "fair_pe_training_last_date"] < dates[-1]


def test_expected_return_is_exact_multiplicative_decomposition() -> None:
    result = horizon_return_components(
        annual_dividend_yield=0.03,
        annual_earnings_growth=0.06,
        fair_pe=12.0,
        current_normalized_pe=11.0,
        annual_cash_rate=0.015,
        horizon_days=242,
    )
    expected = (
        (1.0 + result["expected_dividend_return"])
        * (1.0 + result["expected_earnings_growth_return"])
        * (1.0 + result["expected_rerating_return"])
        - 1.0
    )
    assert result["rerating_convergence"] == pytest.approx(0.5)
    assert result["expected_total_return"] == pytest.approx(expected)
    assert result["expected_excess_total_return"] == pytest.approx(
        expected - result["cash_return"]
    )


def test_edge_groups_are_formed_only_on_realized_rows() -> None:
    count = 12
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=count, freq="ME"),
            "expected_excess_total_return_60d": np.arange(count, dtype=float),
            "realized_excess_total_return_60d": [*np.arange(10, dtype=float), np.nan, np.nan],
        }
    )
    result = evaluate_expected_edge(panel, 60)
    assert result["observations"] == 10
    assert [row["observations"] for row in result["quintiles"]] == [2, 2, 2, 2, 2]
    assert result["spearman_ic"] == pytest.approx(1.0)
