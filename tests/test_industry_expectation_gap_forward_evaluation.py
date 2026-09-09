from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.industry_expectation_gap_forward_evaluation import (
    EvaluationRules,
    ForwardEvaluationError,
    build_forward_evaluation_report,
    determine_observation_state,
    load_trading_calendar,
    resolve_maturity_schedule,
    score_etf_horizon,
    score_fundamental_horizon,
    score_price_horizon,
    validate_evaluation_config,
    verify_hash_manifest,
)


def _rules() -> EvaluationRules:
    return EvaluationRules(
        horizons_trading_days=(60, 120),
        minimum_index_endpoint_weight_coverage=0.98,
        minimum_industry_endpoint_weight_coverage_ratio=0.95,
        minimum_fundamental_core_coverage_ratio=0.95,
        missing_component_reweighting_allowed=False,
        partial_horizon_return_output_allowed=False,
        endpoint_carry_forward_allowed=False,
        balanced_gap_scored=False,
        unobserved_gap_scored=False,
        mixed_fundamental_scored=False,
        minimum_mature_prediction_origins_for_calibration=20,
        minimum_mature_prediction_origins_for_model_comparison=40,
        append_only_outputs=True,
    )


def _config() -> dict:
    return {
        "version": "EVAL_V1",
        "status": "FROZEN_BEFORE_FIRST_FORWARD_OBSERVATION",
        "asset": "510300",
        "benchmark": "000300",
        "prediction_date": "2026-08-18",
        "information_cutoff": "2026-08-18T16:30:00+08:00",
        "entry_date": "2026-08-19",
        "frozen_prediction": {"weight_snapshot_date": "2026-07-31"},
        "rules": {
            **_rules().__dict__,
            "horizons_trading_days": [60, 120],
        },
        "fundamental_confirmation_columns": [
            "weighted_ttm_profit_growth_yoy",
            "weighted_ttm_revenue_growth_yoy",
            "weighted_ttm_roe",
        ],
        "safety": {
            "may_upgrade_original_no_view": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }


def _calendar(start: str = "2026-08-19", periods: int = 130) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _forecast_rows() -> list[dict]:
    return [
        {
            "industry_l1": "行业甲",
            "sector_weight": 0.6,
            "expectation_gap": "POSITIVE",
            "fundamental_60d": "IMPROVEMENT",
            "fundamental_120d": "IMPROVEMENT",
        },
        {
            "industry_l1": "行业乙",
            "sector_weight": 0.4,
            "expectation_gap": "NEGATIVE",
            "fundamental_60d": "DETERIORATION",
            "fundamental_120d": "DETERIORATION",
        },
    ]


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "con_code": ["A.SH", "B.SH"],
            "trade_date": pd.to_datetime(["2026-07-31", "2026-07-31"]),
            "weight": [60.0, 40.0],
        }
    )


def _intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A.SH", "B.SH"],
            "l1_name": ["行业甲", "行业乙"],
            "in_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "out_date": pd.to_datetime([None, None]),
        }
    )


def _constituent_prices(entry: str, maturity: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([entry, entry, maturity, maturity]),
            "con_code": ["A.SH", "B.SH", "A.SH", "B.SH"],
            "total_return_open": [10.0, 20.0, 12.0, 18.0],
            "total_return_close": [10.0, 20.0, 12.0, 18.0],
        }
    )


def _sector_panel(endpoint_date: str) -> pd.DataFrame:
    rows = []
    for snapshot, values in [
        (
            "2026-07-31",
            {
                "行业甲": (0.10, 0.08, 0.10),
                "行业乙": (0.20, 0.15, 0.15),
            },
        ),
        (
            endpoint_date,
            {
                "行业甲": (0.20, 0.10, 0.11),
                "行业乙": (0.10, 0.14, 0.13),
            },
        ),
    ]:
        for industry, (profit, revenue, roe) in values.items():
            rows.append(
                {
                    "date": pd.Timestamp(snapshot),
                    "industry_l1": industry,
                    "sector_weight": 0.6 if industry == "行业甲" else 0.4,
                    "core_feature_coverage_ratio": 1.0,
                    "weighted_ttm_profit_growth_yoy": profit,
                    "weighted_ttm_revenue_growth_yoy": revenue,
                    "weighted_ttm_roe": roe,
                }
            )
    return pd.DataFrame(rows)


def test_config_rejects_partial_horizon_output() -> None:
    config = _config()
    config["rules"]["partial_horizon_return_output_allowed"] = True
    with pytest.raises(ForwardEvaluationError, match="禁止启用"):
        validate_evaluation_config(config)


def test_schedule_counts_entry_as_first_trading_day() -> None:
    calendar = _calendar(periods=70)
    result = resolve_maturity_schedule(
        calendar,
        "2026-08-19",
        [60, 120],
        {"60": calendar[59].date().isoformat(), "120": None},
    )
    assert result[0]["maturity_date"] == calendar[59].date().isoformat()
    assert result[0]["calendar_state"] == "RESOLVED"
    assert result[1]["calendar_state"] == "UNRESOLVED_CALENDAR"


def test_observation_waits_for_entry_without_partial_returns() -> None:
    calendar = _calendar(periods=130)
    schedule = resolve_maturity_schedule(calendar, "2026-08-19", [60, 120])
    constituent = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-18", "2026-08-18"]),
            "con_code": ["A.SH", "B.SH"],
        }
    )
    etf = pd.DataFrame({"date": pd.to_datetime(["2026-08-18"])})
    result = determine_observation_state(
        "2026-08-18", "2026-08-19", schedule, calendar, constituent, etf
    )
    assert result["observation_date"] == "2026-08-18"
    assert result["horizons"][0]["state"] == "WAITING_FOR_ENTRY"
    assert result["horizons"][0]["observed_common_forward_trading_days"] == 0


def test_observation_accumulates_dates_but_does_not_score_early() -> None:
    calendar = _calendar(periods=130)
    schedule = resolve_maturity_schedule(calendar, "2026-08-19", [60, 120])
    observed = calendar[:10]
    constituent = pd.DataFrame(
        [(date, code) for date in observed for code in ("A.SH", "B.SH")],
        columns=["date", "con_code"],
    )
    etf = pd.DataFrame({"date": observed})
    result = determine_observation_state(
        "2026-08-18", "2026-08-19", schedule, calendar, constituent, etf
    )
    assert result["horizons"][0]["state"] == "ACCUMULATING_NO_PEEK"
    assert result["horizons"][0]["observed_common_forward_trading_days"] == 10


def test_price_score_uses_fixed_weights_and_relative_direction() -> None:
    calendar = _calendar(periods=60)
    maturity = calendar[-1].date().isoformat()
    result = score_price_horizon(
        60,
        "2026-08-19",
        maturity,
        "2026-07-31",
        _weights(),
        _intervals(),
        _constituent_prices("2026-08-19", maturity),
        _forecast_rows(),
        _rules(),
    )
    assert result["status"] == "MATURE_SCORED"
    assert result["fixed_basket_return"] == pytest.approx(0.08)
    assert result["direction_accuracy"] == pytest.approx(1.0)
    by_industry = {row["industry_l1"]: row for row in result["industry_rows"]}
    assert by_industry["行业甲"]["relative_return"] == pytest.approx(0.12)
    assert by_industry["行业乙"]["relative_return"] == pytest.approx(-0.18)


def test_price_score_fails_closed_when_index_endpoint_coverage_is_low() -> None:
    calendar = _calendar(periods=60)
    maturity = calendar[-1].date().isoformat()
    prices = _constituent_prices("2026-08-19", maturity)
    prices = prices.loc[~((prices["date"].eq(pd.Timestamp(maturity))) & (prices["con_code"].eq("B.SH")))]
    result = score_price_horizon(
        60,
        "2026-08-19",
        maturity,
        "2026-07-31",
        _weights(),
        _intervals(),
        prices,
        _forecast_rows(),
        _rules(),
    )
    assert result["status"] == "MATURE_DATA_INCOMPLETE"
    assert result["endpoint_weight_coverage"] == pytest.approx(0.6)


def test_fundamental_score_uses_majority_direction_votes() -> None:
    result = score_fundamental_horizon(
        60,
        "2026-11-18",
        "2026-07-31",
        _sector_panel("2026-10-31"),
        _forecast_rows(),
        [
            "weighted_ttm_profit_growth_yoy",
            "weighted_ttm_revenue_growth_yoy",
            "weighted_ttm_roe",
        ],
        _rules(),
    )
    assert result["status"] == "MATURE_SCORED"
    assert result["direction_accuracy"] == pytest.approx(1.0)
    by_industry = {row["industry_l1"]: row for row in result["industry_rows"]}
    assert by_industry["行业甲"]["observed_fundamental"] == "IMPROVEMENT"
    assert by_industry["行业乙"]["observed_fundamental"] == "DETERIORATION"


def test_etf_score_starts_at_entry_open_and_compounds_later_total_returns() -> None:
    calendar = pd.bdate_range("2026-08-19", periods=3)
    etf = pd.DataFrame(
        {
            "date": calendar,
            "open": [100.0, 101.0, 103.0],
            "close": [102.0, 103.0, 104.0],
            "total_return": [0.03, 0.02, 0.01],
        }
    )
    result = score_etf_horizon(3, calendar[0], calendar[-1], calendar, etf)
    assert result["status"] == "MATURE_SCORED"
    assert result["gross_total_return"] == pytest.approx(1.02 * 1.02 * 1.01 - 1.0)
    assert result["transaction_cost_applied"] is False


def test_report_before_maturity_contains_no_partial_return_fields() -> None:
    calendar = _calendar(periods=130)
    schedule = resolve_maturity_schedule(calendar, "2026-08-19", [60, 120])
    constituent = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-18", "2026-08-18"]),
            "con_code": ["A.SH", "B.SH"],
        }
    )
    etf = pd.DataFrame({"date": pd.to_datetime(["2026-08-18"])})
    observation = determine_observation_state(
        "2026-08-18", "2026-08-19", schedule, calendar, constituent, etf
    )
    frozen_result = {
        "model_version": "FORECAST_V1",
        "final_state": "NO_VIEW",
        "industry_aggregation": {"state": "BALANCED_NO_EDGE"},
        "industry_rows": _forecast_rows(),
    }
    report = build_forward_evaluation_report(
        _config(),
        _rules(),
        _config()["safety"],
        calendar,
        observation,
        frozen_result,
        _weights(),
        _intervals(),
        constituent,
        _sector_panel("2026-10-31"),
        etf,
        runtime_sources={},
    )
    assert report["primary_state"] == "WAITING_FOR_ENTRY"
    assert report["horizons"][0]["price_evaluation"] == {
        "status": "WAITING_FOR_ENTRY",
        "partial_return_output": False,
    }
    assert report["may_upgrade_original_no_view"] is False


def test_manifest_verification_detects_frozen_file_change(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("原始内容", encoding="utf-8")
    import hashlib
    import json

    original_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frozen_files": [
                    {"path": "frozen.txt", "sha256": original_hash}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    frozen.write_text("被修改", encoding="utf-8")
    with pytest.raises(ForwardEvaluationError, match="哈希不一致"):
        verify_hash_manifest(tmp_path, "manifest.json")
