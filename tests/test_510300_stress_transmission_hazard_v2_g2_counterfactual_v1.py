"""V2 反事实 G2 V1 的时钟、模型与权限测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.stress_transmission_hazard_v2_g2_counterfactual_v1 import (
    B1_FEATURES,
    B2_FEATURES,
    G2CounterfactualError,
    build_g2_counterfactual_artifacts,
    event_year_block_bootstrap,
    fit_nonnegative_logistic,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g2_counterfactual_v1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


ERAS = [
    {"id": "ERA_1_2015_2017", "start": "2015-01-05", "end": "2017-12-29"},
    {"id": "ERA_2_2018_2020", "start": "2018-01-02", "end": "2020-12-31"},
    {"id": "ERA_3_2021_2023", "start": "2021-01-04", "end": "2023-12-29"},
    {"id": "ERA_4_2024_CUTOFF", "start": "2024-01-02", "end": "2026-08-14"},
]


def _row(
    *,
    origin: str,
    bad10: int,
    event_id: str,
    sequence: int,
    positive_weight: float = 0.5,
) -> dict[str, object]:
    date = pd.Timestamp(origin)
    risk = 0.85 if bad10 else 0.15 + 0.01 * (sequence % 5)
    f_value = 0.80 if bad10 else 0.25
    t_value = 0.88 if bad10 else 0.20 + 0.01 * (sequence % 3)
    return {
        "origin_date": date,
        "horizon_end_date": date + pd.Timedelta(days=14),
        "bad10": bad10,
        "event_id": event_id,
        "sample_group_id": event_id if bad10 else f"NEGATIVE_{date:%Y%m%d}_{sequence}",
        "bootstrap_event_block_id": event_id if bad10 else "",
        "calendar_year_block_id": f"CALENDAR_YEAR_{date.year}",
        "b1_realized_vol20_risk_percentile": risk,
        "b1_drawdown20_risk_percentile": risk - 0.02,
        "b1_downside_return5_risk_percentile": risk + 0.02,
        "F": f_value,
        "T": t_value,
        "T_x_F": t_value * f_value,
        "b2_vs_b1_eligible": True,
        "b2_vs_b1_model_weight": positive_weight if bad10 else 1.0,
    }


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    sequence = 0
    for year in range(2017, 2026):
        for month in (2, 4, 6, 8, 10, 12):
            rows.append(
                _row(
                    origin=f"{year}-{month:02d}-10",
                    bad10=0,
                    event_id="",
                    sequence=sequence,
                )
            )
            sequence += 1
        event_id = f"BAD10_V2_EVENT_TEST_{year}"
        if year == 2023:
            positive_dates = ["2023-12-20", "2024-01-03"]
        else:
            positive_dates = [f"{year}-09-05", f"{year}-09-06"]
        for origin in positive_dates:
            rows.append(
                _row(
                    origin=origin,
                    bad10=1,
                    event_id=event_id,
                    sequence=sequence,
                )
            )
            sequence += 1
        event_rows.append(
            {
                "event_id": event_id,
                "total_positive_origin_count": 2,
                "b2_eligible_positive_origin_count": 2,
                "b2_identifiable_event": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def _build_synthetic(**overrides: object):
    samples, events = _synthetic_inputs()
    arguments: dict[str, object] = {
        "samples": samples,
        "events": events,
        "historical_cutoff": "2026-08-14",
        "era_contracts": ERAS,
    }
    arguments.update(overrides)
    return build_g2_counterfactual_artifacts(**arguments)


def test_fixed_logistic_has_nonnegative_slopes() -> None:
    features = np.asarray([[0.1], [0.2], [0.8], [0.9]], dtype=float)
    outcome = np.asarray([0, 0, 1, 1], dtype=int)
    weight = np.ones(4, dtype=float)
    fitted = fit_nonnegative_logistic(features, outcome, weight)
    assert fitted.converged
    assert fitted.slopes[0] >= 0.0
    with pytest.raises(G2CounterfactualError, match="L2"):
        fit_nonnegative_logistic(features, outcome, weight, l2_penalty=0.5)


def test_prequential_predictions_use_only_label_matured_history() -> None:
    artifacts = _build_synthetic()
    allowed = artifacts.model_vintages.loc[
        artifacts.model_vintages["model_state"].eq(
            "VIEW_ALLOWED_LABEL_MATURED_PREQUENTIAL"
        )
    ]
    assert (
        allowed["training_max_horizon_end_date"]
        < allowed["model_vintage_start"]
    ).all()
    assert artifacts.result["future_calendar_year_training_used"] is False
    assert artifacts.result["unmatured_label_training_used"] is False
    assert artifacts.result["observation_after_historical_cutoff_read"] is False
    assert artifacts.result["formal_g2_status"] == (
        "NOT_ADMISSIBLE_ACTUAL_G1_REMAINS_NO_VIEW"
    )
    assert artifacts.result["formal_model_admitted"] is False


def test_cross_year_positive_event_uses_one_vintage_and_one_era() -> None:
    artifacts = _build_synthetic()
    event = artifacts.predictions.loc[
        artifacts.predictions["event_id"].eq("BAD10_V2_EVENT_TEST_2023")
    ]
    assert len(event) == 2
    assert event["evaluation_vintage_year"].nunique() == 1
    assert int(event["evaluation_vintage_year"].iloc[0]) == 2023
    assert event["era_id"].nunique() == 1
    assert event["era_id"].iloc[0] == "ERA_3_2021_2023"
    assert event["sample_weight"].sum() == pytest.approx(1.0)


def test_b1_and_b2_are_scored_on_exact_same_rows() -> None:
    artifacts = _build_synthetic()
    predictions = artifacts.predictions
    assert predictions[["probability_B1", "probability_B2"]].notna().all().all()
    assert predictions["probability_B1"].between(0.0, 1.0).all()
    assert predictions["probability_B2"].between(0.0, 1.0).all()
    slope_columns = [
        *[f"b1_slope_{column}" for column in B1_FEATURES],
        *[f"b2_slope_{column}" for column in B2_FEATURES],
    ]
    assert (
        artifacts.model_vintages[slope_columns].dropna().to_numpy(dtype=float)
        >= -1.0e-12
    ).all()


def test_bootstrap_is_deterministic() -> None:
    artifacts = _build_synthetic()
    first = event_year_block_bootstrap(
        artifacts.predictions, repetitions=5000, seed=20260903
    )
    second = event_year_block_bootstrap(
        artifacts.predictions, repetitions=5000, seed=20260903
    )
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(G2CounterfactualError, match="5000/20260903"):
        event_year_block_bootstrap(
            artifacts.predictions, repetitions=4999, seed=20260903
        )


def test_cutoff_and_forbidden_path_values_fail_closed() -> None:
    samples, events = _synthetic_inputs()
    samples.loc[samples.index[-1], "horizon_end_date"] = pd.Timestamp("2026-08-15")
    with pytest.raises(G2CounterfactualError, match="历史截止日之后"):
        build_g2_counterfactual_artifacts(
            samples=samples,
            events=events,
            historical_cutoff="2026-08-14",
            era_contracts=ERAS,
        )
    samples, events = _synthetic_inputs()
    samples["minimum_path_return"] = -0.05
    with pytest.raises(G2CounterfactualError, match="禁止读取列"):
        build_g2_counterfactual_artifacts(
            samples=samples,
            events=events,
            historical_cutoff="2026-08-14",
            era_contracts=ERAS,
        )


def test_actual_g1_is_never_promoted_by_counterfactual_numeric_result() -> None:
    artifacts = _build_synthetic()
    assert artifacts.result["actual_G1_DATA_AND_EVENTS"] == (
        "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    )
    assert artifacts.result["assumed_G1_DATA_AND_EVENTS"] == (
        "PASS_BY_USER_COUNTERFACTUAL_ASSUMPTION"
    )
    assert artifacts.result["g3_formally_authorized"] is False
    assert artifacts.result["probability_threshold_selected"] is False
    assert artifacts.result["portfolio_or_performance_artifact_read"] is False
    assert artifacts.result["position_impact"] == 0


def test_current_config_is_valid_and_manifest_verifies_if_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    manifest_path = Path(config["freeze_contract"]["manifest_output"])
    if manifest_path.is_file():
        verify_frozen_manifest(DEFAULT_CONFIG)
