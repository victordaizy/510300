"""沪深 A 股复合低风险 V2 冻结规则与证据完整性测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/a_share_sh_sz_low_risk_regime_entry_v2.yaml"
EXPECTED_BUCKET2_MANIFEST_SHA256 = (
    "5194175a32ba8c64ac8ef2b8ebf8e9ebb0586c5d04e9abe28b9101399345ce6a"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_scope_is_explicitly_shanghai_and_shenzhen_only(contract: dict) -> None:
    assert contract["protocol"]["model_id"] == "A_SHARE_SH_SZ_LOW_RISK_REGIME_ENTRY_V2"
    assert contract["protocol"]["initial_status"] == "DISCOVERY_ONLY"
    assert contract["protocol"]["current_view_status"] == "NO_VIEW"
    assert contract["universe"]["type"] == "SH_SZ_A_POINT_IN_TIME"
    assert contract["universe"]["exchanges"] == ["SSE", "SZSE"]
    assert contract["universe"]["explicitly_out_of_scope_exchanges"] == ["BSE"]
    assert not (ROOT / "config/a_share_all_a_low_risk_regime_entry_v2.yaml").exists()


def test_point_in_time_inputs_require_available_at_and_shares(contract: dict) -> None:
    inputs = contract["inputs"]
    for key in (
        "security_master_history",
        "daily_market",
        "share_count_history",
        "security_status_intervals",
        "industry_l1_intervals",
        "trading_calendar",
    ):
        assert "available_at" in inputs[key]["required_columns"]
        assert inputs[key]["path"].startswith(
            "data/raw/a_share_sh_sz_low_risk_regime_entry_v2/"
        )
    share_columns = set(inputs["share_count_history"]["required_columns"])
    assert {"total_a_shares", "tradable_a_shares", "effective_date"} <= share_columns
    assert contract["data_contract"]["price_for_market_cap"] == "RAW_CLOSE"
    assert contract["data_contract"]["price_for_risk_and_trend"] == (
        "TOTAL_RETURN_CLOSE"
    )
    assert contract["data_contract"]["current_share_count_history_backfill_forbidden"]
    assert contract["data_contract"]["current_survivor_universe_backfill_forbidden"]


def test_market_cap_and_liquidity_thresholds_are_frozen(contract: dict) -> None:
    market_cap = contract["market_cap"]
    assert market_cap["smoothing"] == "MEDIAN_20_MARKET_TRADING_DAYS"
    assert market_cap["minimum_valid_observations"] == 20
    assert market_cap["entry"] == {
        "min_total_mcap_cny": 10_000_000_000,
        "min_float_mcap_cny": 5_000_000_000,
        "min_float_mcap_cross_section_percentile": 0.30,
    }
    assert market_cap["holding"] == {
        "min_total_mcap_cny": 8_000_000_000,
        "min_float_mcap_cny": 4_000_000_000,
        "min_float_mcap_cross_section_percentile": 0.25,
        "exit_confirmation_reviews": 2,
    }
    liquidity = contract["liquidity"]
    assert liquidity["min_median_amount_20_cny"] == 100_000_000
    assert liquidity["min_valid_trading_days_60"] == 58
    assert liquidity["max_suspension_days_20"] == 0
    assert liquidity["max_zero_volume_days_20"] == 0
    assert liquidity["return_based_threshold_change_forbidden"]


def test_low_risk_regime_and_trend_rules_are_separate(contract: dict) -> None:
    risk = contract["risk_score"]
    assert list(risk["metrics"]) == [
        "rv60",
        "rv120",
        "downside_vol60",
        "max_drawdown120_magnitude",
    ]
    assert [item["weight"] for item in risk["metrics"].values()] == [
        0.25,
        0.25,
        0.25,
        0.25,
    ]
    assert risk["entry_max"] == 0.20
    assert risk["holding_max"] == 0.40
    assert risk["market_cap_in_score_forbidden"]
    assert risk["trend_in_score_forbidden"]

    regime = contract["volatility_regime"]
    assert regime["history_lookback_market_days"] == 504
    assert regime["history_excludes_current_observation"]
    assert regime["min_valid_history"] == 252
    assert regime["entry_max_percentile"] == 0.40
    assert regime["exit_min_percentile"] == 0.70

    trend = contract["trend_veto"]
    assert trend["confirmed_downtrend_requires_both"]
    assert trend["exit_confirmation_reviews"] == 2
    assert trend["momentum_expression"] == (
        "TOTAL_RETURN_CLOSE_T_MINUS_5_DIV_T_MINUS_60_MINUS_1"
    )


def test_portfolio_benchmarks_costs_and_governance_are_frozen(contract: dict) -> None:
    portfolio = contract["research_portfolio"]
    assert portfolio["target_names"] == 20
    assert portfolio["target_weight_per_name"] == 0.05
    assert portfolio["max_names_per_industry_l1"] == 3
    assert portfolio["insufficient_candidates"] == "HOLD_CASH"
    assert not portfolio["relax_thresholds"]

    wrapper = contract["small_account_wrapper"]
    assert wrapper["capital_cny"] == 20_000
    assert wrapper["target_names"] == 3
    assert wrapper["lot_size_shares"] == 100
    assert wrapper["minimum_trade_notional_cny"] == 5_000
    assert wrapper["maximum_one_lot_notional_cny"] == 7_000
    assert not wrapper["fixed_nominal_price_filter"]

    assert contract["benchmarks"]["required"] == [
        "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT_TOTAL_RETURN",
        "ELIGIBLE_UNIVERSE_FLOAT_MCAP_WEIGHTED_TOTAL_RETURN",
        "H00300_TOTAL_RETURN",
        "CNY20000_510300_BUY_AND_HOLD_ACCOUNT",
    ]
    assert contract["costs"]["commission_rate_per_leg"] == 0.0003
    assert contract["costs"]["minimum_commission_cny_per_leg"] == 5

    governance = contract["governance"]
    assert governance["old_bucket2_modification"] == "FORBIDDEN"
    assert governance["paper_signal"] == "DISABLED"
    assert governance["position_mapping"] == "DISABLED"
    assert governance["order_generation"] == "DISABLED"
    assert governance["broker_connection"] == "DISABLED"
    assert governance["automatic_order"] == "DISABLED"
    assert governance["live_trading"] == "NOT_AUTHORIZED"


def test_bucket2_permanent_freeze_is_preserved(contract: dict) -> None:
    boundary = contract["research_boundary"]
    path = ROOT / boundary["parent_bucket2_evidence_manifest"]
    assert boundary["parent_bucket2_terminal_status"] == (
        "STRICT_BUCKET2_CODE_AND_TIME_HOLDOUT_REJECTED_FROZEN"
    )
    assert not boundary["reuse_bucket2_results_for_tuning"]
    assert not boundary["reopen_bucket2_analysis"]
    assert sha256_file(path) == EXPECTED_BUCKET2_MANIFEST_SHA256
    assert boundary["parent_bucket2_evidence_manifest_sha256"] == (
        EXPECTED_BUCKET2_MANIFEST_SHA256
    )


def test_data_readiness_report_never_contains_a_return_view(contract: dict) -> None:
    path = ROOT / contract["paths"]["data_readiness_json"]
    if not path.exists():
        pytest.skip("冻结脚本尚未生成数据可行性报告")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["model_id"] == contract["protocol"]["model_id"]
    assert report["view_status"] == "NO_VIEW"
    assert report["scope"]["included_exchanges"] == ["SSE", "SZSE"]
    assert report["scope"]["explicitly_out_of_scope_exchanges"] == ["BSE"]
    assert not report["return_values_read"]
    assert not report["return_metrics_computed"]
    assert not report["selection_generated"]
    assert not report["position_mapping_generated"]
    assert not report["orders_generated"]


def test_protocol_manifest_hashes_are_immutable_when_present(contract: dict) -> None:
    path = ROOT / contract["paths"]["protocol_manifest"]
    if not path.exists():
        pytest.skip("冻结脚本尚未生成协议清单")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == contract["protocol"]["model_id"]
    assert manifest["included_exchanges"] == ["SSE", "SZSE"]
    assert manifest["explicitly_out_of_scope_exchanges"] == ["BSE"]
    assert not manifest["return_values_read_before_freeze"]
    assert not manifest["return_metrics_computed_before_freeze"]
    assert not manifest["eligible_universe_audit_run_before_freeze"]
    assert not manifest["bucket2_result_reopened"]
    assert not manifest["bucket2_result_used_for_tuning"]
    assert not manifest["position_mapping_generated"]
    assert not manifest["orders_generated"]
    assert not manifest["broker_connection_enabled"]
    assert not manifest["live_trading_authorized"]
    for relative, expected in manifest["frozen_files"].items():
        assert sha256_file(ROOT / relative) == expected, relative
