"""沪深 A 股三只极端低风险确认策略 V1 的冻结规则测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1.yaml"
EXPECTED_BUCKET2_MANIFEST_SHA256 = (
    "5194175a32ba8c64ac8ef2b8ebf8e9ebb0586c5d04e9abe28b9101399345ce6a"
)
EXPECTED_SHARED_AUDITOR_SHA256 = (
    "4edc523db0b9c85e1d0fa9fb02abb9bcc1cbc848377c1243ff7fd6cb1e0b4c42"
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


def test_concentrated_model_is_independent_and_sh_sz_only(contract: dict) -> None:
    assert contract["protocol"]["model_id"] == "A_SHARE_HS_CONCENTRATED_LOW_RISK_TREND_V1"
    assert contract["protocol"]["initial_status"] == "DISCOVERY_ONLY"
    assert contract["protocol"]["current_view_status"] == "NO_VIEW"
    assert contract["universe"]["exchanges"] == ["SSE", "SZSE"]
    assert contract["universe"]["included_boards"] == [
        "SSE_MAIN",
        "SSE_STAR",
        "SZSE_MAIN",
        "SZSE_CHINEXT",
    ]
    assert contract["research_boundary"]["independent_project_id_required"]
    assert contract["research_boundary"]["shared_raw_data_contract"] == (
        "A_SHARE_SH_SZ_POINT_IN_TIME_RAW_V1"
    )


def test_only_unfrozen_draft_threshold_is_changed_to_ten_percent(contract: dict) -> None:
    change = contract["protocol"]["draft_change"]
    assert change["volatility_regime_entry_max_percentile_before_freeze"] == 0.30
    assert change["volatility_regime_entry_max_percentile_frozen"] == 0.10
    regime = contract["volatility_regime"]
    assert regime["entry_max_percentile"] == 0.10
    assert regime["normal_exit_percentile"] == 0.70
    assert regime["emergency_exit_percentile"] == 0.90
    assert regime["emergency_rv20_div_rv120"] == 1.60
    assert regime["lookback_days"] == 504
    assert regime["min_valid_history"] == 252
    assert regime["history_excludes_current_observation"]
    assert regime["percentile_method"] == "EMPIRICAL_MIDRANK_CDF"


def test_market_cap_liquidity_and_five_risk_metrics(contract: dict) -> None:
    assert contract["market_cap"]["entry"] == {
        "min_total_mcap_20d_median_cny": 10_000_000_000,
        "min_float_mcap_20d_median_cny": 5_000_000_000,
        "min_float_mcap_percentile": 0.30,
    }
    assert contract["market_cap"]["holding"] == {
        "min_total_mcap_20d_median_cny": 8_000_000_000,
        "min_float_mcap_20d_median_cny": 4_000_000_000,
        "min_float_mcap_percentile": 0.25,
        "confirmation_reviews": 2,
    }
    assert contract["liquidity"] == {
        "min_amount_20d_median_cny": 100_000_000,
        "max_suspension_days_20": 0,
        "max_zero_volume_days_20": 0,
        "min_valid_returns_120": 115,
        "signal_date_must_be_tradable": True,
        "return_based_threshold_change_forbidden": True,
    }
    metrics = contract["risk_score"]["metrics"]
    assert list(metrics) == [
        "rv60",
        "rv120",
        "downside_vol60",
        "max_drawdown120",
        "parkinson_range_vol60",
    ]
    assert [item["weight"] for item in metrics.values()] == [0.20] * 5
    assert contract["risk_score"]["entry_current_max"] == 0.12
    assert contract["risk_score"]["entry_previous_20d_max"] == 0.20
    assert contract["risk_score"]["entry_previous_40d_max"] == 0.25
    assert contract["risk_score"]["stable_score_max"] == 0.18
    assert contract["risk_score"]["worst_component_rank_max"] == 0.35
    assert contract["risk_score"]["holding_max"] == 0.35


def test_stability_selection_cash_ladder_and_five_day_risk_review(contract: dict) -> None:
    assert contract["risk_score_stability"] == {
        "current_weight": 0.50,
        "previous_20d_weight": 0.30,
        "previous_40d_weight": 0.20,
        "history_must_be_point_in_time": True,
    }
    assert contract["selection"]["target_names"] == 3
    assert contract["selection"]["max_names_per_industry"] == 1
    assert contract["selection"]["max_pairwise_correlation_120d"] == 0.80
    assert contract["selection"]["pairwise_rule"] == "STRICTLY_LESS_THAN"
    assert contract["selection"]["insufficient_candidates"] == "HOLD_CASH"
    assert contract["review"]["full_selection_frequency_trading_days"] == 20
    assert contract["review"]["risk_review_frequency_trading_days"] == 5
    assert contract["review"]["minimum_holding_days_for_normal_replacement"] == 20
    assert not contract["review"]["risk_review_can_select_replacement"]
    assert contract["portfolio"] == {
        "three_names": {"target_weight_each": 0.30, "cash_weight": 0.10},
        "two_names": {"target_weight_each": 0.40, "cash_weight": 0.20},
        "one_name": {"maximum_weight": 0.40, "cash_weight": 0.60},
        "zero_names": {"cash_weight": 1.0},
    }


def test_execution_and_governance_do_not_authorize_trading(contract: dict) -> None:
    assert contract["execution"]["commission_rate_per_leg"] == 0.0003
    assert contract["execution"]["minimum_commission_cny_per_leg"] == 5
    assert contract["execution"]["sell_stamp_duty_rate"] == 0.0005
    assert contract["execution"]["one_way_slippage_rate"] == 0.0005
    assert contract["small_account"]["initial_capital_cny"] == 20_000
    assert contract["small_account"]["lot_size"] == 100
    assert contract["small_account"]["minimum_trade_notional_cny"] == 5_000
    assert contract["small_account"]["maximum_one_lot_equity_weight"] == 0.40
    governance = contract["governance"]
    assert governance["old_bucket2_modification"] == "FORBIDDEN"
    assert governance["modify_510300_position"] is False
    assert governance["paper_signal"] == "DISABLED"
    assert governance["position_mapping"] == "DISABLED"
    assert governance["order_generation"] == "DISABLED"
    assert governance["broker_connection"] == "DISABLED"
    assert governance["automatic_order"] == "DISABLED"
    assert governance["live_trading"] == "NOT_AUTHORIZED"


def test_old_bucket2_and_shared_auditor_hashes_are_preserved(contract: dict) -> None:
    boundary = contract["research_boundary"]
    bucket2 = ROOT / boundary["parent_bucket2_evidence_manifest"]
    shared = ROOT / "scripts/audit_a_share_sh_sz_low_risk_regime_entry_v2_inputs.py"
    assert sha256_file(bucket2) == EXPECTED_BUCKET2_MANIFEST_SHA256
    assert boundary["parent_bucket2_evidence_manifest_sha256"] == (
        EXPECTED_BUCKET2_MANIFEST_SHA256
    )
    assert sha256_file(shared) == EXPECTED_SHARED_AUDITOR_SHA256
    assert not boundary["reuse_bucket2_results_for_tuning"]
    assert not boundary["reopen_bucket2_analysis"]


def test_data_readiness_report_is_no_view_and_return_free(contract: dict) -> None:
    path = ROOT / contract["paths"]["data_readiness_json"]
    if not path.exists():
        pytest.skip("集中模型冻结脚本尚未生成数据审计报告")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["model_id"] == contract["protocol"]["model_id"]
    assert report["view_status"] == "NO_VIEW"
    assert report["scope"]["included_exchanges"] == ["SSE", "SZSE"]
    assert not report["return_values_read"]
    assert not report["return_metrics_computed"]
    assert not report["selection_generated"]
    assert not report["position_mapping_generated"]
    assert not report["orders_generated"]


def test_protocol_manifest_hashes_are_immutable_when_present(contract: dict) -> None:
    path = ROOT / contract["paths"]["protocol_manifest"]
    if not path.exists():
        pytest.skip("集中模型冻结脚本尚未生成协议清单")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == contract["protocol"]["model_id"]
    assert manifest["decision_model"] == "TOP3"
    assert manifest["frozen_entry_ts_vol_pct"] == 0.10
    assert manifest["draft_entry_ts_vol_pct"] == 0.30
    assert manifest["view_status"] == "NO_VIEW"
    assert not manifest["return_values_read_before_freeze"]
    assert not manifest["return_metrics_computed_before_freeze"]
    assert not manifest["selection_generated"]
    assert not manifest["top1_shadow_started"]
    assert not manifest["bucket2_result_reopened"]
    assert not manifest["bucket2_result_used_for_tuning"]
    assert not manifest["position_mapping_generated"]
    assert not manifest["orders_generated"]
    assert not manifest["broker_connection_enabled"]
    assert not manifest["live_trading_authorized"]
    for relative, expected in manifest["frozen_files"].items():
        assert sha256_file(ROOT / relative) == expected, relative
