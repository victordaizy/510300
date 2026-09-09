"""510300 成分脆弱性 DSV5 增量检验 V1 的冻结回归测试。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.freeze_510300_constituent_fragility_dsv5_increment_v1 import (
    build_lineage_audit,
)

from research.constituent_fragility_dsv5_increment_v1 import (
    B1_FEATURES,
    B2_FEATURES,
    DSV5ProtocolError,
    build_b1_daily_features,
    build_dsv5_label_ledger,
    build_origin_schedule,
    circular_block_bootstrap_lower_bound,
    evaluate_g2,
    evaluate_g3,
    fit_nonnegative_qlike_ridge,
    generate_b0_b1_predictions,
    prepare_parent_features,
    qlike_deviance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "config/510300_constituent_fragility_dsv5_increment_v1.yaml"


def _protocol() -> dict:
    return {
        "scope": {"historical_cutoff_for_features": "2025-12-31"},
        "inputs": {
            "parent_internal_features": {
                "required_columns": [
                    "date",
                    "f1_breadth_risk_percentile",
                    "f2_leadership_risk_percentile",
                    "f3_comovement_risk_percentile",
                    "F",
                    "t1_breadth_drop_risk_percentile",
                    "t2_tail_diffusion_risk_percentile",
                    "t3_comovement_accel_risk_percentile",
                    "internal_feature_state",
                    "four_state_daily_coverage_state",
                    "return20_coverage_ratio",
                    "tail_coverage_ratio",
                    "comovement_scoreable_member_ratio",
                ]
            }
        },
        "label": {
            "horizon_trading_days": 5,
            "annualization_days": 252,
            "qlike_epsilon": 1.0e-8,
        },
        "origin_schedule": {
            "cadence_trading_days": 5,
            "registered_offsets": [0, 1, 2, 3, 4],
            "primary_offset": 0,
            "first_training_origins": 10,
            "minimum_primary_total_origins": 25,
            "minimum_primary_prequential_evaluation_origins": 15,
            "evaluation_eras": 3,
            "minimum_origins_each_era": 5,
            "insufficient_state": "NO_VIEW_INSUFFICIENT_NONOVERLAPPING_ORIGINS",
        },
        "models": {"refit_every_evaluation_origins": 13, "l2_lambda": 1.0},
        "evaluation": {
            "bootstrap": {
                "block_length_origins": 13,
                "repetitions": 250,
                "one_sided_confidence": 0.90,
                "random_seed": 20260905,
            }
        },
        "gates": {
            "G2_B1_VS_B0": {"minimum_positive_eras": 2},
            "G3_B2_VS_B1": {
                "overall_qlike_relative_improvement_minimum": 0.02,
                "minimum_positive_eras": 2,
                "minimum_positive_registered_offsets": 4,
                "highest_to_lowest_actual_dsv5_quintile_ratio_minimum": 1.5,
                "mean_predicted_to_actual_ratio_minimum": 0.80,
                "mean_predicted_to_actual_ratio_maximum": 1.25,
            },
        },
    }


def _parent_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    sequence = np.linspace(0.1, 0.9, len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "f1_breadth_risk_percentile": sequence,
            "f2_leadership_risk_percentile": np.clip(sequence + 0.05, 0, 1),
            "f3_comovement_risk_percentile": np.clip(sequence - 0.05, 0, 1),
            "t1_breadth_drop_risk_percentile": sequence[::-1],
            "t2_tail_diffusion_risk_percentile": np.clip(sequence[::-1] + 0.03, 0, 1),
            "t3_comovement_accel_risk_percentile": np.clip(sequence[::-1] - 0.03, 0, 1),
            "internal_feature_state": "VIEW_ALLOWED",
            "four_state_daily_coverage_state": "VIEW_ALLOWED",
            "return20_coverage_ratio": 1.0,
            "tail_coverage_ratio": 1.0,
            "comovement_scoreable_member_ratio": 1.0,
        }
    )
    frame["F"] = frame[
        [
            "f1_breadth_risk_percentile",
            "f2_leadership_risk_percentile",
            "f3_comovement_risk_percentile",
        ]
    ].median(axis=1)
    return frame


def test_frozen_protocol_is_prediction_only_and_uses_final_review_precedence() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol"]["model_id"] == "510300_CONSTITUENT_FRAGILITY_DSV5_INCREMENT_V1"
    assert protocol["protocol"]["new_alpha_family"] is False
    assert protocol["protocol"]["new_estimand"] is True
    assert protocol["source_adjudication"]["precedence"] == (
        "FINAL_REVIEW_SUPERSEDES_INITIAL_PROPOSAL_ON_CONFLICT"
    )
    assert protocol["source_adjudication"]["resolved_conflicts"]["causal_percentile_warmup"] == (
        "REUSE_PARENT_PERSISTED_DEFINITION_WITHOUT_NEW_252_OBSERVATION_OVERLAY"
    )
    assert protocol["governance"]["portfolio_evaluation"] == "NOT_ALLOWED"
    assert protocol["governance"]["sharpe_calculation"] == "NOT_ALLOWED"
    assert protocol["governance"]["position_impact"] == 0


def test_parent_f_and_tc_are_medians_and_macro_t4_is_absent() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prepared = prepare_parent_features(_parent_frame(dates), _protocol())
    expected_tc = prepared[
        [
            "t1_breadth_drop_risk_percentile",
            "t2_tail_diffusion_risk_percentile",
            "t3_comovement_accel_risk_percentile",
        ]
    ].median(axis=1)
    assert np.allclose(prepared["T_C"], expected_tc)
    assert np.allclose(prepared["T_C_X_F"], prepared["T_C"] * prepared["F"])
    assert "t4_funding_shock_risk_percentile" not in prepared.columns


def test_parent_f_identity_mismatch_is_rejected() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    frame = _parent_frame(dates)
    frame.loc[1, "F"] += 0.1
    with pytest.raises(DSV5ProtocolError, match="持久化F"):
        prepare_parent_features(frame, _protocol())


def test_b1_features_follow_frozen_formulas() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = 100.0 * np.exp(np.linspace(0.0, 0.12, len(dates)))
    etf = pd.DataFrame(
        {"date": dates, "open": close * 0.999, "close": close, "symbol": "510300.SH"}
    )
    dividends = pd.DataFrame(
        columns=["symbol", "record_date", "ex_date", "payment_date", "cash_dividend_per_share", "source"]
    )
    built = build_b1_daily_features(etf, dividends, annualization_days=252)
    log_returns = np.diff(np.log(close))
    expected_rv20 = math.sqrt(252.0 / 20.0 * float(np.square(log_returns[-20:]).sum()))
    assert built.iloc[-1]["LOG_RV20"] == pytest.approx(math.log(expected_rv20))
    assert built.iloc[-1]["NEG5"] == pytest.approx(0.0)
    assert built.iloc[-1]["DD20"] == pytest.approx(0.0)
    assert bool(built.iloc[-1]["B1_VALID"])


def test_origin_grid_is_fixed_five_day_spacing_and_eras_are_near_equal() -> None:
    dates = pd.bdate_range("2024-01-01", periods=180)
    parent = prepare_parent_features(_parent_frame(dates), _protocol())
    b1 = pd.DataFrame({"date": dates, "B1_VALID": True})
    schedule, metrics = build_origin_schedule(
        parent_features=parent,
        b1_daily=b1,
        protocol=_protocol(),
    )
    assert metrics["passed"] is True
    primary = schedule.loc[(schedule["offset"].eq(0)) & schedule["eligible"]]
    positions = pd.Series(np.arange(len(dates)), index=dates)
    steps = np.diff([positions.loc[date] for date in primary["origin_date"]])
    assert set(steps.tolist()) == {5}
    assert primary.iloc[0]["sample_role"] == "INITIAL_TRAINING"
    era_counts = list(metrics["primary_era_counts"].values())
    assert max(era_counts) - min(era_counts) <= 1


def test_origin_no_view_does_not_shift_or_reanchor_grid() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    frame = _parent_frame(dates)
    frame.loc[25, "internal_feature_state"] = "NO_VIEW"
    parent = prepare_parent_features(frame, _protocol())
    b1 = pd.DataFrame({"date": dates, "B1_VALID": True})
    schedule, _ = build_origin_schedule(
        parent_features=parent,
        b1_daily=b1,
        protocol=_protocol(),
    )
    primary = schedule.loc[schedule["offset"].eq(0)].reset_index(drop=True)
    assert primary.loc[5, "eligible"] == False  # noqa: E712
    assert primary.loc[5, "origin_date"] == dates[25]
    assert primary.loc[6, "origin_date"] == dates[30]


def test_dsv5_excludes_preentry_entitlement_and_includes_later_entitlement() -> None:
    dates = pd.bdate_range("2024-01-01", periods=7)
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 7,
            "close": [100.0, 90.0, 100.0, 90.0, 100.0, 100.0, 100.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH", "510300.SH"],
            "record_date": [dates[0], dates[2]],
            "ex_date": [dates[1], dates[3]],
            "payment_date": [dates[2], dates[4]],
            "cash_dividend_per_share": [10.0, 10.0],
            "source": ["测试", "测试"],
        }
    )
    schedule = pd.DataFrame(
        {
            "offset": [0],
            "origin_date": [dates[0]],
            "entry_date": [dates[1]],
            "horizon_end_date": [dates[5]],
            "eligible": [True],
            "sample_role": ["INITIAL_TRAINING"],
            "era_id": [""],
        }
    )
    ledger = build_dsv5_label_ledger(
        schedule=schedule,
        b1_daily=daily,
        dividends=dividends,
        protocol=_protocol(),
    )
    expected = 252.0 / 5.0 * math.log(0.9) ** 2
    assert ledger.iloc[0]["DSV5"] == pytest.approx(expected)
    assert ledger.iloc[0]["entitled_cash_dividend_per_share_in_horizon"] == pytest.approx(10.0)


def test_nonnegative_qlike_ridge_never_returns_negative_slopes() -> None:
    x = np.linspace(-1.0, 1.0, 120)
    frame = pd.DataFrame(
        {
            "x_positive": x,
            "x_inverse": -x,
            "target": np.exp(-2.0 + 0.8 * x),
        }
    )
    fitted = fit_nonnegative_qlike_ridge(
        frame,
        feature_names=["x_positive", "x_inverse"],
        target_name="target",
        l2_lambda=1.0,
        epsilon=1.0e-8,
    )
    assert (fitted.coefficients >= 0.0).all()
    assert fitted.coefficients[0] > fitted.coefficients[1]


def test_qlike_deviance_is_zero_for_perfect_positive_forecast() -> None:
    actual = pd.Series([0.01, 0.02, 0.03])
    predicted = actual + 1.0e-8
    loss = qlike_deviance(actual, predicted, epsilon=1.0e-8)
    assert np.allclose(loss, 0.0, atol=1.0e-14)


def test_block_bootstrap_is_deterministic() -> None:
    differences = np.linspace(-0.1, 0.2, 80)
    first = circular_block_bootstrap_lower_bound(
        differences,
        block_length=13,
        repetitions=300,
        lower_quantile=0.10,
        random_seed=20260905,
    )
    second = circular_block_bootstrap_lower_bound(
        differences,
        block_length=13,
        repetitions=300,
        lower_quantile=0.10,
        random_seed=20260905,
    )
    assert first == second


def _prediction_frame(*, include_b2: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for offset in range(5):
        for index in range(60):
            actual = 0.002 + 0.0004 * index
            row = {
                "offset": offset,
                "origin_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index * 7 + offset),
                "era_id": f"ERA_{index // 20 + 1}" if offset == 0 else "ROBUSTNESS_OFFSET",
                "DSV5": actual,
                "predicted_B0": actual * 3.0,
                "predicted_B1": actual * 2.0,
            }
            if include_b2:
                row["predicted_B2"] = actual
            rows.append(row)
    return pd.DataFrame(rows)


def test_g2_passes_only_when_b1_beats_b0_in_latest_era() -> None:
    result = evaluate_g2(_prediction_frame(include_b2=False), _protocol())
    assert result["passed"] is True
    reversed_frame = _prediction_frame(include_b2=False)
    reversed_frame["predicted_B1"] = reversed_frame["DSV5"] * 4.0
    failed = evaluate_g2(reversed_frame, _protocol())
    assert failed["passed"] is False


def test_g3_all_atomic_gates_pass_for_strictly_better_calibrated_b2() -> None:
    result = evaluate_g3(_prediction_frame(include_b2=True), _protocol())
    assert result["passed"] is True
    assert result["positive_offset_count"] == 5
    assert all(result["checks"].values())


def test_prequential_training_uses_only_mature_labels() -> None:
    protocol = _protocol()
    protocol["origin_schedule"]["registered_offsets"] = [0]
    protocol["origin_schedule"]["first_training_origins"] = 20
    origins = pd.bdate_range("2020-01-03", periods=70, freq="5B")
    x = np.linspace(0.0, 1.0, len(origins))
    panel = pd.DataFrame(
        {
            "offset": 0,
            "origin_date": origins,
            "entry_date": origins + pd.offsets.BDay(1),
            "horizon_end_date": origins + pd.offsets.BDay(5),
            "era_id": [f"ERA_{min(3, index // 17 + 1)}" for index in range(len(origins))],
            "DSV5": np.exp(-3.0 + 0.2 * x),
            "LOG_RV20": x,
            "NEG5": x / 2.0,
            "DD20": x / 3.0,
            "T_C": 1.0 - x,
            "T_C_X_F": (1.0 - x) * 0.5,
            "F": 0.5,
        }
    )
    predictions, coefficients = generate_b0_b1_predictions(panel, protocol=protocol)
    assert len(predictions) == 50
    assert predictions["maximum_training_horizon_end_at_refit"].le(
        predictions["origin_date"]
    ).all()
    b1_rows = coefficients.loc[coefficients["model"].eq("B1")]
    assert b1_rows.loc[~b1_rows["feature"].eq("INTERCEPT"), "coefficient"].ge(0.0).all()


def test_output_contract_contains_no_portfolio_artifact_paths() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output_paths = [str(value).lower() for value in protocol["outputs"].values()]
    forbidden = ("portfolio", "nav", "trade", "order", "sharpe")
    assert not any(token in path for path in output_paths for token in forbidden)
    assert tuple(protocol["models"]["B1"]["inputs"]) == B1_FEATURES
    assert tuple(protocol["models"]["B2"]["inputs"]) == B2_FEATURES


def test_actual_lineage_audit_finds_no_exact_estimand_match() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    audit = build_lineage_audit(PROJECT_ROOT, protocol)
    assert audit["status"] == "PASS_LINEAGE_NO_EXACT_DSV5_B1_TO_B2_INCREMENT_TEST_FOUND"
    assert audit["new_orthogonal_alpha_family"] is False
    assert all(item["decision"] != "EXACT_MATCH" for item in audit["comparisons"])
