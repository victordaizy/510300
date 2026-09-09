"""行业预期差预测原点簇成熟度测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from research.industry_expectation_gap_origin_maturity import (
    OriginMaturityError,
    build_origin_cluster_maturity_status,
)


def _report(
    prediction_date: str,
    *,
    mature: bool,
    industry_rows: int = 25,
    observation_date: str | None = None,
) -> dict:
    prediction = pd.Timestamp(prediction_date)
    entry = prediction + pd.offsets.BDay(1)
    maturity = entry + pd.offsets.BDay(59)
    rows = [{"industry_l1": f"行业{index}"} for index in range(industry_rows)]
    return {
        "evaluation_version": "INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1",
        "prediction_date": prediction.date().isoformat(),
        "observation_date": observation_date or (
            maturity.date().isoformat() if mature else entry.date().isoformat()
        ),
        "entry_date": entry.date().isoformat(),
        "original_prediction_state": "NO_VIEW",
        "may_upgrade_original_no_view": False,
        "horizons": [
            {
                "horizon_trading_days": 60,
                "maturity_date": maturity.date().isoformat(),
                "state": "MATURE_SCORED" if mature else "ACCUMULATING_NO_PEEK",
                "price_evaluation": {"industry_rows": rows},
            }
        ],
    }


def test_one_date_with_25_industry_rows_is_one_origin_cluster() -> None:
    result = build_origin_cluster_maturity_status(
        [_report("2026-08-18", mature=True, industry_rows=25)]
    )
    assert result["origin_cluster_count"] == 1
    assert result["mature_origin_cluster_count"] == 1
    assert result["origin_clusters"][0]["origin_cluster"] == "2026-08-18"
    assert result["origin_clusters"][0]["industry_row_count"] == 25
    assert result["industry_rows_count_as_time_samples"] is False
    assert result["status"] == "COLLECTING_FORWARD"


def test_20_mature_origins_still_require_four_non_overlapping_blocks() -> None:
    reports = [
        _report(str(pd.Timestamp("2026-01-02") + pd.offsets.BDay(index)), mature=True)
        for index in range(20)
    ]
    result = build_origin_cluster_maturity_status(reports)
    assert result["mature_origin_cluster_count"] == 20
    assert result["non_overlapping_60d_block_count"] < 4
    assert result["calibration"]["eligible"] is False


def test_spaced_origins_meet_calibration_but_not_model_comparison() -> None:
    reports = [
        _report(
            str(pd.Timestamp("2020-01-02") + pd.offsets.BDay(index * 65)),
            mature=True,
        )
        for index in range(20)
    ]
    result = build_origin_cluster_maturity_status(reports)
    assert result["non_overlapping_60d_block_count"] >= 4
    assert result["calibration"]["eligible"] is True
    assert result["model_comparison"]["eligible"] is False
    assert result["status"] == "CALIBRATION_ELIGIBLE_NOT_MODEL_COMPARISON"


def test_original_no_view_cannot_be_rewritten() -> None:
    report = _report("2026-08-18", mature=False)
    report["original_prediction_state"] = "VIEW"
    with pytest.raises(OriginMaturityError, match="NO_VIEW"):
        build_origin_cluster_maturity_status([report])


def test_append_only_reports_for_one_origin_select_latest_without_double_count() -> None:
    waiting = _report(
        "2026-08-18",
        mature=False,
        observation_date="2026-08-18",
    )
    accumulating = _report(
        "2026-08-18",
        mature=False,
        observation_date="2026-08-19",
    )
    result = build_origin_cluster_maturity_status([waiting, accumulating])
    assert result["origin_cluster_count"] == 1
    assert result["append_only_report_count"] == 2
    cluster = result["origin_clusters"][0]
    assert cluster["first_observation_date"] == "2026-08-18"
    assert cluster["latest_observation_date"] == "2026-08-19"
    assert cluster["append_only_report_count"] == 2


def test_same_origin_and_observation_date_conflict_fails_closed() -> None:
    first = _report("2026-08-18", mature=False, observation_date="2026-08-19")
    conflicting = _report(
        "2026-08-18",
        mature=False,
        industry_rows=3,
        observation_date="2026-08-19",
    )
    with pytest.raises(OriginMaturityError, match="observation_date"):
        build_origin_cluster_maturity_status([first, conflicting])


def test_origin_state_regression_fails_closed() -> None:
    mature = _report("2026-08-18", mature=True, observation_date="2026-11-18")
    regressed = _report(
        "2026-08-18",
        mature=False,
        observation_date="2026-11-19",
    )
    with pytest.raises(OriginMaturityError, match="倒退"):
        build_origin_cluster_maturity_status([mature, regressed])
