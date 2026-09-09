from __future__ import annotations

import pandas as pd
import pytest

from research.stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1 import (
    INPUT_READ_COLUMNS,
    G1BIdentifiabilityError,
    audit_prequential_identifiability,
    build_quarterly_vintages,
    parse_eras,
    prepare_allowed_sample,
)


def _fixed_eras() -> list[dict[str, object]]:
    return [
        {
            "id": "ERA_1_2015_2017",
            "label": "2015-2017",
            "start": "2015-01-05",
            "end": "2017-12-29",
            "g1a_identifiable_event_count": 0,
            "logical_prequential_upper_bound": 0,
        },
        {
            "id": "ERA_2_2018_2020",
            "label": "2018-2020",
            "start": "2018-01-02",
            "end": "2020-12-31",
            "g1a_identifiable_event_count": 4,
            "logical_prequential_upper_bound": 4,
        },
        {
            "id": "ERA_3_2021_2023",
            "label": "2021-2023",
            "start": "2021-01-04",
            "end": "2023-12-29",
            "g1a_identifiable_event_count": 0,
            "logical_prequential_upper_bound": 19,
        },
        {
            "id": "ERA_4_2024_CUTOFF",
            "label": "2024-2026",
            "start": "2024-01-02",
            "end": "2026-08-14",
            "g1a_identifiable_event_count": 0,
            "logical_prequential_upper_bound": 7,
        },
    ]


def _sample() -> pd.DataFrame:
    rows = [
        ("2018-01-02", "2018-01-16", 0, None, True, "YEAR_2018"),
        ("2018-02-01", "2018-02-15", 1, "E1", True, "YEAR_2018"),
        ("2018-02-02", "2018-02-16", 1, "E1", True, "YEAR_2018"),
        ("2018-04-02", "2018-04-16", 0, None, True, "YEAR_2018"),
        ("2018-04-10", "2018-04-24", 1, "E2", True, "YEAR_2018"),
        ("2018-06-29", "2018-07-13", 1, "E3", True, "YEAR_2018"),
        ("2018-07-02", "2018-07-16", 1, "E3", True, "YEAR_2018"),
        ("2018-07-20", "2018-08-03", 1, "E4", True, "YEAR_2018"),
    ]
    return pd.DataFrame(
        {
            "origin_date": [row[0] for row in rows],
            "horizon_end_date": [row[1] for row in rows],
            "bad10": [row[2] for row in rows],
            "event_id": [row[3] for row in rows],
            "b2_vs_b1_eligible": [row[4] for row in rows],
            "split_assignment": ["UNASSIGNED_PRE_MODEL_FREEZE"] * len(rows),
            "calendar_year_block_id": [row[5] for row in rows],
        },
        columns=list(INPUT_READ_COLUMNS),
    )


def _expected_g1a() -> dict[str, object]:
    return {
        "total_sample_day_count": 8,
        "total_positive_origin_count": 6,
        "total_non_event_risk_day_count": 2,
        "total_independent_event_count": 4,
        "b2_common_sample_day_count": 8,
        "b2_eligible_non_event_risk_day_count": 2,
        "b2_identifiable_event_count": 4,
        "event_era_distribution": {
            "ERA_1_2015_2017": 0,
            "ERA_2_2018_2020": 4,
            "ERA_3_2021_2023": 0,
            "ERA_4_2024_CUTOFF": 0,
        },
    }


def _audit(frame: pd.DataFrame | None = None):
    return audit_prequential_identifiability(
        _sample() if frame is None else frame,
        fixed_eras=_fixed_eras(),
        expected_g1a=_expected_g1a(),
        minimum_training_positive_events=1,
        minimum_training_negative_risk_days=1,
        minimum_prequential_events_per_era=5,
        minimum_qualified_eras=3,
    )


def test_quarterly_vintage_is_first_sample_trading_day_after_quarter_end() -> None:
    sample = prepare_allowed_sample(_sample())
    vintages = build_quarterly_vintages(sample["origin_date"])

    assert list(vintages["quarter_id"]) == ["2017Q4", "2018Q1", "2018Q2"]
    assert [value.date().isoformat() for value in vintages["model_vintage_date"]] == [
        "2018-01-02",
        "2018-04-02",
        "2018-07-02",
    ]


def test_first_positive_event_is_training_history_not_prequential_evaluation() -> None:
    artifacts = _audit()
    events = artifacts.event_audit.set_index("event_id")

    assert not bool(events.loc["E1", "prequential_predicted"])
    assert events.loc["E1", "prediction_state"] == (
        "NO_VIEW_INSUFFICIENT_LABEL_MATURED_TRAINING_CLASSES"
    )
    assert bool(events.loc["E2", "prequential_predicted"])
    assert int(events.loc["E2", "training_positive_event_count"]) == 1
    assert int(events.loc["E2", "training_negative_risk_day_count"]) == 1


def test_event_crossing_quarter_boundary_stays_on_one_older_vintage() -> None:
    artifacts = _audit()
    events = artifacts.event_audit.set_index("event_id")
    vintages = artifacts.vintage_audit.set_index("quarter_id")

    assert events.loc["E3", "model_vintage_date"] == pd.Timestamp("2018-04-02")
    assert bool(events.loc["E3", "no_event_split"])
    assert not bool(events.loc["E3", "event_present_in_own_training"])
    assert int(vintages.loc["2018Q2", "training_positive_event_count"]) == 2


def test_audit_counts_only_prequential_events_and_never_fits_model() -> None:
    artifacts = _audit()

    assert artifacts.result["prequential_predicted_independent_event_count"] == 3
    assert artifacts.result["prequential_events_by_era"] == {
        "ERA_1_2015_2017": 0,
        "ERA_2_2018_2020": 3,
        "ERA_3_2021_2023": 0,
        "ERA_4_2024_CUTOFF": 0,
    }
    assert artifacts.result["qualified_era_count"] == 0
    assert artifacts.result["logical_max_qualified_eras"] == 2
    assert artifacts.result["G1B_PREQUENTIAL_ERA_IDENTIFIABILITY"] == (
        "NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS"
    )
    assert artifacts.result["model_trained"] is False
    assert artifacts.result["return_evaluation"] == "NOT_ALLOWED"


def test_extra_or_forbidden_source_column_is_rejected() -> None:
    frame = _sample()
    frame["T"] = 0.5
    with pytest.raises(G1BIdentifiabilityError, match="精确等于冻结顺序"):
        prepare_allowed_sample(frame)


def test_parent_split_must_remain_unassigned() -> None:
    frame = _sample()
    frame.loc[0, "split_assignment"] = "TRAIN"
    with pytest.raises(G1BIdentifiabilityError, match="必须全部保持未分配"):
        prepare_allowed_sample(frame)


def test_event_enters_training_only_after_whole_event_horizon_matures() -> None:
    artifacts = _audit()
    vintage = artifacts.vintage_audit.set_index("quarter_id").loc["2018Q2"]

    assert vintage["model_vintage_date"] == pd.Timestamp("2018-07-02")
    assert int(vintage["training_positive_event_count"]) == 2
    assert bool(vintage["no_future_training"])
    assert bool(vintage["b1_and_b2_training_identifiable"])


def test_frozen_era_boundary_change_is_rejected() -> None:
    eras = _fixed_eras()
    eras[1] = {**eras[1], "start": "2018-01-03"}
    with pytest.raises(G1BIdentifiabilityError, match="冻结时代发生变化"):
        parse_eras(eras)


def test_five_event_and_three_era_gates_cannot_be_lowered() -> None:
    with pytest.raises(G1BIdentifiabilityError, match="每时代前序事件门必须保持 5"):
        audit_prequential_identifiability(
            _sample(),
            fixed_eras=_fixed_eras(),
            expected_g1a=_expected_g1a(),
            minimum_training_positive_events=1,
            minimum_training_negative_risk_days=1,
            minimum_prequential_events_per_era=4,
            minimum_qualified_eras=3,
        )
    with pytest.raises(G1BIdentifiabilityError, match="合格时代门必须保持 3"):
        audit_prequential_identifiability(
            _sample(),
            fixed_eras=_fixed_eras(),
            expected_g1a=_expected_g1a(),
            minimum_training_positive_events=1,
            minimum_training_negative_risk_days=1,
            minimum_prequential_events_per_era=5,
            minimum_qualified_eras=2,
        )

