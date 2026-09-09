"""510300宏观压力规避2015起点事后窗口敏感性的冻结前测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import macro_stress_avoidance_2015_v2 as sensitivity
from research.macro_stress_avoidance_v1 import _calendar_trailing_quantile
from scripts.download_510300_macro_stress_inputs_2015_v2 import (
    _pbc_search_titles,
)


ROOT = Path(__file__).resolve().parents[1]


def test_inherited_protocol_changes_scope_but_preserves_rules_and_assets() -> None:
    config = sensitivity.load_config()
    assert config["protocol"]["study_id"] == sensitivity.STUDY_ID
    assert config["protocol"]["data_start"] == "2015-01-01"
    assert config["protocol"]["evidence_class"] == (
        "POST_REJECTION_WINDOW_SENSITIVITY_ONLY"
    )
    assert config["protocol"]["allowed_assets"] == ["510300.SH", "CASH_CNY"]
    assert config["protocol"]["main_trailing_years"] == 5
    assert config["protocol"]["robustness_trailing_years"] == 3
    assert config["protocol"]["percentile"] == pytest.approx(0.90)
    assert config["protocol"]["no_parameter_rescue"] is True
    assert config["point_in_time"]["trailing_window_type"] == (
        "EXACT_CALENDAR_YEARS"
    )
    assert config["point_in_time"]["expanding_window_forbidden"] is True
    assert config["point_in_time"]["shortened_window_forbidden"] is True
    assert config["factor_rules"]["liquidity"]["formula"] == (
        "MEDIAN_5_FDR007_MINUS_MEDIAN_60_FDR007"
    )
    assert config["factor_rules"]["score"]["raw_target_mapping"] == {
        "0": 1.0,
        "1": 1.0,
        "2": 0.5,
        "3": 0.0,
        "4": 0.0,
    }
    assert config["event_study"]["minimum_independent_events"] == 8
    assert config["historical_gates"]["five_year_net_sharpe_minimum"] == 1.20


def test_post_rejection_evidence_cannot_override_rejection_or_authorize_trade() -> None:
    config = sensitivity.load_config()
    governance = config["sensitivity_governance"]
    assert governance["original_2021_rejection_overridden"] is False
    assert governance["original_2021_rejection_rescue_forbidden"] is True
    assert governance["eligible_to_claim_preregistered_confirmation"] is False
    assert governance["eligible_to_authorize_paper_or_live_trading"] is False
    assert config["governance"]["position_mapping_enabled"] is False
    assert config["governance"]["order_generation_enabled"] is False
    assert config["governance"]["broker_connection_enabled"] is False
    assert config["governance"]["live_trading_authorized"] is False


def test_prelaunch_and_premonthly_factors_are_left_censored_without_proxy() -> None:
    config = sensitivity.load_config()
    governance = config["sensitivity_governance"]
    assert governance["exact_2015_factor_history_available"] is False
    assert governance["fdr007_official_launch_month"] == "2017-05"
    assert governance["fdr007_first_official_valid_date"] == "2017-05-31"
    assert governance["pre_launch_fdr007_handling"] == (
        "LEFT_CENSORED_NOT_ZERO_NOT_INTERPOLATED_NO_PROXY"
    )
    assert governance["tsf_stock_yoy_2015_official_frequency"] == "QUARTERLY"
    assert governance["tsf_stock_yoy_first_monthly_reference_period"] == (
        "2016-01"
    )
    assert governance["pre_monthly_tsf_handling"] == (
        "LEFT_CENSORED_NO_QUARTERLY_TO_MONTHLY_EXPANSION_"
        "NO_INTERPOLATION_NO_PROXY"
    )


def test_calendar_quantile_waits_five_years_from_first_valid_derived_value() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2017-05-31",
                "2017-08-23",
                "2022-05-31",
                "2022-08-22",
                "2022-08-23",
            ]
        )
    )
    values = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0])
    threshold, eligible = _calendar_trailing_quantile(
        dates, values, years=5, quantile=0.90
    )
    assert eligible.tolist() == [False, False, False, False, True]
    assert threshold.iloc[-1] == pytest.approx(2.8)


def test_macro_input_audit_preserves_structural_left_censoring_and_extreme_pmi() -> None:
    audit_path = ROOT / "reports" / "data_quality" / (
        "510300_macro_stress_inputs_2015_v2.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["proxy_substitution_used"] is False
    fdr = audit["fdr007_left_censoring"]
    assert fdr["first_official_valid_date"] == "2017-05-31"
    assert fdr["pre_launch_placeholder_rows"] == 599
    assert fdr["zero_fill_used"] is False
    assert fdr["interpolation_used"] is False
    assert fdr["FR007_proxy_used"] is False
    tsf = audit["tsf_stock_yoy_left_censoring"]
    assert tsf["first_monthly_reference_period"] == "2016-01"
    assert "NO_INTERPOLATION_NO_PROXY" in tsf["handling"]
    assert audit["series"]["pmi_new_orders"]["first"] == "2015-01"
    assert audit["series"]["pmi_new_orders"]["minimum"] == pytest.approx(29.3)
    assert audit["series"]["tsf_stock_yoy"]["first"] == "2016-01"
    for series in audit["series"].values():
        assert series["duplicate_keys"] == 0
        assert series["missing_first_release_values"] == 0


def test_macro_output_files_match_audit_scope_and_have_no_missing_values() -> None:
    config = sensitivity.load_config()
    contracts = config["data_contracts"]
    fdr = pd.read_parquet(ROOT / contracts["fdr007"]["file"])
    fx = pd.read_parquet(ROOT / contracts["usdcny_midpoint"]["file"])
    pmi = pd.read_parquet(ROOT / contracts["pmi_new_orders"]["file"])
    tsf = pd.read_parquet(ROOT / contracts["tsf_stock_yoy"]["file"])
    assert str(pd.Timestamp(fdr["date"].min()).date()) == "2017-05-31"
    assert str(pd.Timestamp(fx["date"].min()).date()) == "2015-01-05"
    assert pmi["reference_period"].iloc[0] == "2015-01"
    assert tsf["reference_period"].iloc[0] == "2016-01"
    for frame in (fdr, fx, pmi, tsf):
        assert frame["first_release_value"].notna().all()


def test_market_input_audit_is_exact_on_frozen_overlap() -> None:
    audit_path = ROOT / "reports" / "data_quality" / (
        "510300_market_inputs_2015_v2.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["missing_h00300_dates_on_510300_calendar"] == 0
    for overlap in audit["overlap"].values():
        assert overlap["passed"] is True
        for field in overlap["fields"].values():
            assert field["maximum_absolute_difference"] == pytest.approx(0.0)
            assert field["mismatches_above_1e_minus_9"] == 0
    for output in audit["outputs"].values():
        assert output["duplicate_dates"] == 0


def test_pbc_search_titles_cover_month_end_and_period_end_variants() -> None:
    march = _pbc_search_titles(2017, 3)
    june = _pbc_search_titles(2017, 6)
    december = _pbc_search_titles(2017, 12)
    assert "2017年3月末社会融资规模存量统计数据报告" in march
    assert "2017年一季度社会融资规模存量统计数据报告" in march
    assert "2017年上半年社会融资规模存量统计数据报告" in june
    assert "2017年社会融资规模存量统计数据报告" in december


def test_sensitivity_wrapper_forces_goal_false_even_if_mechanism_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = sensitivity.load_config()
    base_report = {
        "decision": "HISTORICAL_CANDIDATE_PASS",
        "data_scope": {},
        "portfolio_backtest": {
            "main_gates": {"five_year_net_sharpe_at_least_1_20": True}
        },
    }
    monkeypatch.setattr(
        sensitivity.base,
        "evaluate_protocol",
        lambda _: (base_report.copy(), {"point_in_time_panel": pd.DataFrame()}),
    )
    report, _ = sensitivity.evaluate_protocol(config)
    assert report["decision"] == (
        "POST_REJECTION_WINDOW_SENSITIVITY_HISTORICAL_GATES_PASS_"
        "NO_CANDIDATE_OR_TRADE_AUTHORIZATION"
    )
    assert report["mechanism_decision_before_evidence_class"] == (
        "HISTORICAL_CANDIDATE_PASS"
    )
    assert report["historical_sharpe_threshold_met"] is True
    assert report["goal_achieved"] is False
    assert report["predecessor"]["overridden"] is False
    assert report["position_mapping_enabled"] is False
    assert report["order_generation_enabled"] is False
    assert report["broker_connection_enabled"] is False
    assert report["live_trading_authorized"] is False
