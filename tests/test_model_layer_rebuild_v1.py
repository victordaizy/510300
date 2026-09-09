from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.model_layer_rebuild_v1 import (
    FractionalCosts,
    audit_and_load_inputs,
    build_all_technical_targets,
    build_fixed_mix_benchmark,
    build_r5_component_targets,
    build_valuation_engineering_audit,
    load_config,
    multiple_testing_diagnostics,
    simulate_fractional_targets,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()


@pytest.fixture(scope="module")
def audited_inputs() -> tuple[dict[str, pd.DataFrame], dict]:
    return audit_and_load_inputs(ROOT, CONFIG)


@pytest.fixture(scope="module")
def technical_targets(
    audited_inputs: tuple[dict[str, pd.DataFrame], dict]
) -> dict[str, pd.DataFrame]:
    datasets, _ = audited_inputs
    return build_all_technical_targets(datasets, CONFIG)


def test_registry_is_frozen_and_counts_every_trial() -> None:
    identifiers = [item["id"] for item in CONFIG["technical_models"]]
    assert identifiers == [
        "TECH_01_TSMOM_63_126_252",
        "TECH_03_DONCHIAN_55_20",
        "TECH_03_DONCHIAN_55_20_25_FLOOR",
        "TECH_03_DONCHIAN_100_50_ROBUSTNESS",
        "TECH_05_TREND_PULLBACK",
        "TECH_08_BREADTH_TREND",
    ]
    settings = CONFIG["evaluation"]["multiple_testing"]
    assert settings["return_tested_technical_candidate_count"] == 6
    assert settings["registered_technical_trial_count"] == 7
    assert len(CONFIG["valuation_models"]["val01_registered_trials"]) == 6
    assert len(CONFIG["valuation_models"]["val02_registered_trials"]) == 2


def test_real_data_gates_separate_technical_valuation_and_r5_replay(
    audited_inputs: tuple[dict[str, pd.DataFrame], dict]
) -> None:
    datasets, audit = audited_inputs
    assert len(datasets["evaluation_market"]) == 1211
    assert audit["technical_gate"]["status"] == "PASS"
    assert audit["valuation_gate"]["status"] == "NO_VIEW_BLOCKED_NON_VINTAGE_HISTORY"
    assert audit["valuation_gate"]["return_calculation_allowed"] is False
    assert audit["valuation_gate"]["point_in_time_months"] == 60
    assert audit["r5_replay_gate"]["current_constituent_market_cap_non_null_ratio"] == 0.0
    assert audit["r5_replay_gate"]["raw_replay_status"] == "BLOCKED_SOURCE_DRIFT_CURRENT_MARKET_CAP_MISSING"
    assert audit["r5_replay_gate"]["decomposition_input_status"] == "PASS_PRESERVED_DERIVED_SIGNAL_SNAPSHOT"


def test_valuation_formulas_are_audited_but_emit_no_signal_or_return(
    audited_inputs: tuple[dict[str, pd.DataFrame], dict]
) -> None:
    datasets, _ = audited_inputs
    engineered, audit = build_valuation_engineering_audit(datasets, CONFIG)
    assert {"raw_ey", "normalized_ey", "normalized_ey_spread"}.issubset(engineered.columns)
    assert audit["return_calculation_allowed"] is False
    assert audit["signals_or_returns_emitted"] is False
    assert "target_position" not in engineered.columns
    assert audit["registered_val01_trial_count"] == 6
    assert audit["registered_val02_trial_count"] == 2


def test_all_technical_targets_cover_the_same_calendar_and_allowed_grid(
    technical_targets: dict[str, pd.DataFrame],
    audited_inputs: tuple[dict[str, pd.DataFrame], dict],
) -> None:
    datasets, _ = audited_inputs
    expected_calendar = datasets["evaluation_market"]["date"].tolist()
    allowed = set(CONFIG["execution"]["allowed_position_grid"])
    for frame in technical_targets.values():
        assert frame["date"].tolist() == expected_calendar
        assert frame["target_position"].notna().all()
        assert set(frame["target_position"].unique()).issubset(allowed)
        assert frame["risk_off_override"].eq(False).all()


def test_tech01_only_reviews_every_five_ready_days_and_mapping_is_exact(
    technical_targets: dict[str, pd.DataFrame],
) -> None:
    frame = technical_targets["TECH_01_TSMOM_63_126_252"]
    review = frame.loc[frame["trade_allowed"]]
    assert len(review) > 0
    non_initial = review.iloc[1:]
    expected = np.select(
        [
            non_initial["momentum_sign_sum"] <= -2,
            non_initial["momentum_sign_sum"] <= 0,
            non_initial["momentum_sign_sum"] <= 2,
        ],
        [0.0, 0.25, 0.75],
        default=1.0,
    )
    assert np.allclose(non_initial["target_position"], expected)
    source_indices = frame.index[frame["trade_allowed"]].to_numpy()
    assert np.all(np.diff(source_indices[1:]) == 5)


def test_donchian_uses_prior_bars_and_25pct_floor_is_separate(
    technical_targets: dict[str, pd.DataFrame],
) -> None:
    main = technical_targets["TECH_03_DONCHIAN_55_20"]
    entered = main.loc[main["donchian_action"].eq("ENTER")]
    exited = main.loc[main["donchian_action"].eq("EXIT")]
    assert len(entered) > 0 and len(exited) > 0
    assert (entered["signal_close"] > entered["entry_channel"]).all()
    assert (exited["signal_close"] < exited["exit_channel"]).all()
    assert set(main["target_position"].unique()).issubset({0.0, 1.0})
    floor = technical_targets["TECH_03_DONCHIAN_55_20_25_FLOOR"]
    assert set(floor["target_position"].unique()).issubset({0.25, 1.0})
    assert floor["target_position"].min() == 0.25
    assert floor["donchian_action"].tolist() == main["donchian_action"].tolist()


def test_tech05_is_finite_layered_pullback_not_martingale(
    technical_targets: dict[str, pd.DataFrame],
) -> None:
    frame = technical_targets["TECH_05_TREND_PULLBACK"]
    assert frame["target_position"].max() <= 0.50
    first = frame.loc[frame["pullback_action"].eq("ENTER_FIRST_LAYER")]
    second = frame.loc[frame["pullback_action"].eq("ADD_SECOND_LAYER")]
    assert len(first) > 0
    assert first["target_position"].eq(0.25).all()
    assert second["target_position"].eq(0.50).all()
    assert (first["long_trend"] & first["z20"].lt(-1.0)).all()


def test_tech08_uses_point_in_time_breadth_and_exact_mapping(
    technical_targets: dict[str, pd.DataFrame],
) -> None:
    frame = technical_targets["TECH_08_BREADTH_TREND"]
    assert frame["valid_for_direction_model"].all()
    assert frame["ma60_weight_coverage"].min() >= 0.99
    expected = np.select(
        [
            frame["trend_up"] & frame["official_weighted_above_ma60_share"].ge(0.55),
            frame["trend_up"] & frame["official_weighted_above_ma60_share"].ge(0.40),
            frame["trend_up"],
        ],
        [1.0, 0.75, 0.50],
        default=0.0,
    )
    assert np.allclose(frame["target_position"], expected)


def test_fractional_execution_is_next_bar_and_dividend_uses_ex_date_inventory() -> None:
    dates = pd.bdate_range("2026-01-05", periods=5)
    market = pd.DataFrame(
        {"date": dates, "etf_open": [4.0] * 5, "etf_close": [4.0, 4.0, 3.9, 4.0, 4.0]}
    )
    targets = pd.DataFrame(
        {
            "date": dates,
            "target_position": [1.0] * 5,
            "trade_allowed": [True, False, False, False, False],
        }
    )
    dividends = pd.DataFrame(
        {
            "ex_date": [dates[2]],
            "payment_date": [dates[4]],
            "cash_dividend_per_share": [0.1],
        }
    )
    costs = FractionalCosts(0.0, 0.0, 0.0, 242)
    ledger, trades = simulate_fractional_targets(market, dividends, targets, 1.0, costs)
    assert trades.iloc[0]["date"] == dates[1]
    assert trades.iloc[0]["signal_date"] == dates[0]
    entitlement = float(trades.iloc[0]["quantity"]) * 0.1
    assert ledger.loc[2, "dividend_entitlement_today"] == pytest.approx(entitlement)
    assert ledger.loc[4, "dividend_payment_today"] == pytest.approx(entitlement)
    assert ledger.loc[4, "dividend_receivable"] == pytest.approx(0.0)


def test_fixed_mix_benchmarks_are_exact_standard_exposures(
    audited_inputs: tuple[dict[str, pd.DataFrame], dict]
) -> None:
    datasets, _ = audited_inputs
    for exposure in (0.25, 0.50, 0.75, 1.0):
        ledger = build_fixed_mix_benchmark(
            datasets["evaluation_market"], datasets["dividends"], exposure, 1.0, 0.015, 242
        )
        assert ledger["actual_position"].eq(exposure).all()
        assert np.isfinite(ledger["equity"]).all()
        assert (ledger["equity"] > 0).all()


def test_r5_decomposition_has_all_eight_component_masks_without_risk_leakage(
    audited_inputs: tuple[dict[str, pd.DataFrame], dict]
) -> None:
    datasets, _ = audited_inputs
    targets = build_r5_component_targets(datasets, CONFIG, ROOT)
    assert len(targets) == 8
    assert set(targets) == {item["id"] for item in CONFIG["r5_decomposition"]["variants"]}
    assert targets["R5_000_BUY_HOLD"]["target_position"].eq(1.0).all()
    for identifier, frame in targets.items():
        assert len(frame) == 1211
        assert frame["target_position"].between(0.0, 1.0).all(), identifier
    assert targets["R5_010_TREND_ONLY"]["risk_off_override"].eq(False).all()
    assert targets["R5_100_VALUATION_ONLY"]["risk_off_override"].eq(False).all()


def test_multiple_testing_outputs_white_spa_dsr_and_estimable_pbo() -> None:
    rng = np.random.default_rng(20260818)
    dates = pd.bdate_range("2021-01-01", periods=320)
    frames = {}
    for index in range(6):
        frames[f"M{index}"] = pd.DataFrame(
            {"date": dates, "active_return": rng.normal(index * 1e-5, 0.008, len(dates))}
        )
    config = deepcopy(CONFIG)
    config["evaluation"]["multiple_testing"]["repetitions"] = 100
    diagnostics = multiple_testing_diagnostics(frames, config)
    assert 0.0 <= diagnostics["white_reality_check"]["p_value"] <= 1.0
    assert 0.0 <= diagnostics["hansen_spa"]["p_value"] <= 1.0
    assert set(diagnostics["deflated_sharpe_ratio"]) == set(frames)
    assert diagnostics["pbo"]["split_count"] == 70
    assert 0.0 <= diagnostics["pbo"]["value"] <= 1.0


def test_live_and_order_safety_switches_are_disabled() -> None:
    protocol = CONFIG["protocol"]
    assert protocol["true_forward_start"] is None
    assert protocol["position_mapping_enabled"] is False
    assert protocol["order_generation_enabled"] is False
    assert protocol["broker_connection_enabled"] is False
