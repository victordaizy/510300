from __future__ import annotations

import numpy as np
import pandas as pd

from research.structural_signal_power_identifiability_audit_v1 import (
    build_nonoverlap_diagnostics,
    build_paired_loss_panel,
    build_power_requirements,
    build_sharpe_identifiability,
    newey_west_long_run_variance,
    realized_annualized_volatility,
)


def _predictions() -> pd.DataFrame:
    origins = pd.date_range("2021-01-31", periods=48, freq="ME")
    actual = 0.01 + 0.05 * np.sin(np.arange(48) / 4.0)
    rows: list[dict[str, object]] = []
    for model_id, prediction in {
        "STRUCTURE_WITH_INTERACTIONS": actual + 0.01 * np.cos(np.arange(48)),
        "HISTORICAL_MEAN": np.repeat(0.01, 48),
        "PRICE_REALIZED_RISK": actual + 0.02 * np.sin(np.arange(48)),
    }.items():
        for origin, forecast, observed in zip(
            origins, prediction, actual, strict=True
        ):
            rows.append(
                {
                    "origin": origin,
                    "target_id": "EXPECTED_60D_TOTAL_RETURN",
                    "model_id": model_id,
                    "prediction": forecast,
                    "actual": observed,
                    "actual_status": "ACTUAL_OBSERVED",
                }
            )
    return pd.DataFrame(rows)


def test_paired_loss_panel_uses_common_observed_origins() -> None:
    source = _predictions()
    source.loc[
        source["model_id"].eq("HISTORICAL_MEAN")
        & source["origin"].eq(pd.Timestamp("2021-01-31")),
        "prediction",
    ] = np.nan
    paired = build_paired_loss_panel(
        source,
        target_specs={
            "EXPECTED_60D_TOTAL_RETURN": {
                "horizon_market_days": 60,
                "monthly_overlap_block_origins": 3,
            }
        },
        candidate_model="STRUCTURE_WITH_INTERACTIONS",
        baseline_models=["HISTORICAL_MEAN", "PRICE_REALIZED_RISK"],
        evaluation_start="2021-01-01",
        evaluation_end="2024-12-31",
    )

    counts = paired.groupby("baseline_model").size().to_dict()
    assert counts == {"HISTORICAL_MEAN": 47, "PRICE_REALIZED_RISK": 48}
    assert np.allclose(
        paired["loss_gain_vs_baseline"],
        paired["baseline_squared_loss"] - paired["candidate_squared_loss"],
    )


def test_newey_west_long_run_variance_is_finite_and_nonnegative() -> None:
    values = np.array([1.0, 0.5, -0.2, 0.1, 0.3, -0.1])
    result = newey_west_long_run_variance(values, max_lag=2)
    assert np.isfinite(result)
    assert result >= 0.0


def test_power_requirements_are_monotone_and_apply_nonoverlap_bound() -> None:
    paired = build_paired_loss_panel(
        _predictions(),
        target_specs={
            "EXPECTED_60D_TOTAL_RETURN": {
                "horizon_market_days": 60,
                "monthly_overlap_block_origins": 3,
            }
        },
        candidate_model="STRUCTURE_WITH_INTERACTIONS",
        baseline_models=["HISTORICAL_MEAN"],
        evaluation_start="2021-01-01",
        evaluation_end="2024-12-31",
    )
    power = build_power_requirements(
        paired,
        target_specs={
            "EXPECTED_60D_TOTAL_RETURN": {
                "horizon_market_days": 60,
                "monthly_overlap_block_origins": 3,
            }
        },
        relative_effects=[0.01, 0.02, 0.05],
        z_one_sided_alpha=1.6448536269514722,
        z_target_power=0.8416212335729143,
        origins_per_year=12,
        short_horizon_years=5.0,
        acceptable_years=10.0,
    )
    ordered = power.sort_values("hypothetical_relative_mse_improvement")
    assert ordered["required_independent_cycles_iid"].is_monotonic_decreasing
    assert (
        ordered["required_monthly_origins_nonoverlap"]
        == ordered["required_independent_cycles_iid"] * 3
    ).all()
    assert (
        ordered["conservative_required_monthly_origins"]
        >= ordered["required_monthly_origins_nonoverlap"]
    ).all()

    phases = build_nonoverlap_diagnostics(
        paired,
        target_specs={
            "EXPECTED_60D_TOTAL_RETURN": {
                "horizon_market_days": 60,
                "monthly_overlap_block_origins": 3,
            }
        },
    )
    assert len(phases) == 3
    assert set(phases["phase"]) == {0, 1, 2}


def test_realized_volatility_and_sharpe_lower_bound_are_scale_only() -> None:
    dates = pd.bdate_range("2021-01-01", periods=600)
    returns = np.repeat(0.0002, 600) + 0.01 * np.sin(np.arange(600))
    closes = 100.0 * np.cumprod(1.0 + returns)
    volatility = realized_annualized_volatility(
        pd.DataFrame({"date": dates, "close": closes}),
        date_column="date",
        value_column="close",
        start="2021-01-01",
        end="2024-12-31",
        annualization_days=252,
    )
    result = build_sharpe_identifiability(
        reference_sharpe=0.29,
        target_sharpe=1.20,
        realized_volatility=volatility["annualized_volatility"],
        z_one_sided_alpha=1.6448536269514722,
        z_target_power=0.8416212335729143,
        dependence_multipliers=[1.0, 2.0, 3.0],
    )

    assert np.isclose(result["required_sharpe_increment"], 0.91)
    assert result["iid_gaussian_years_using_target_standard_error"] == 13
    assert [
        row["required_calendar_years_lower_bound"]
        for row in result["dependence_scenarios"]
    ] == [13, 26, 39]
    assert result["empirical_sharpe_test_allowed"] is False
    assert result["status"] == (
        "LOWER_BOUND_ONLY_NO_INVESTABLE_STRATEGY_RETURN_SERIES"
    )
