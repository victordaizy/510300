from __future__ import annotations

import copy
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from research.post_close_stale_price_capture_v2 import (
    BUNDLE_STATUS,
    PROJECT_ID,
    NoViewError,
    ProtocolError,
    assert_actual_fill_recording_authorized,
    assert_authoritative_trade_date,
    build_daily_observation,
    build_target_maturity,
    canonical_json_sha256,
    classify_signal,
    conditional_mean_interval,
    evaluate_g0,
    fit_hac_ols,
    intended_capital_return,
    load_config,
    load_json_object,
    round_trip_cost_bps,
    select_frozen_contract,
    select_prior_mature_model_rows,
    select_t1_sell_lot,
    validate_manifest,
    write_model_vintage,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_post_close_stale_price_capture_v2.yaml"
SOURCE_RECEIPT_PATH = ROOT / "config" / "510300_post_close_stale_price_capture_v2_source_receipt.json"
MANIFEST_PATH = ROOT / "config" / "510300_post_close_stale_price_capture_v2_manifest.json"


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


def sample_daily_payload() -> dict:
    return {
        "trade_date": "2026-09-07",
        "collection_started_at": "2026-09-07T14:54:59+08:00",
        "local_ntp_offset_ms": 18.5,
        "contract_candidates": [
            {
                "contract_id": "CN202609",
                "contract_month": "2026-09",
                "expiry_date": "2026-09-29",
                "is_continuous": False,
                "cumulative_volume_0900_1455": 12000,
                "volume_exchange_timestamp": "2026-09-07T14:54:59+08:00",
                "volume_received_at": "2026-09-07T14:54:59.400000+08:00",
            },
            {
                "contract_id": "CN202610",
                "contract_month": "2026-10",
                "expiry_date": "2026-10-29",
                "is_continuous": False,
                "cumulative_volume_0900_1455": 8000,
                "volume_exchange_timestamp": "2026-09-07T14:54:59+08:00",
                "volume_received_at": "2026-09-07T14:54:59.500000+08:00",
            },
            {
                "contract_id": "CN202612",
                "contract_month": "2026-12",
                "expiry_date": "2026-12-29",
                "is_continuous": False,
                "cumulative_volume_0900_1455": 30000,
                "volume_exchange_timestamp": "2026-09-07T14:54:59+08:00",
                "volume_received_at": "2026-09-07T14:54:59.600000+08:00",
            },
        ],
        "selected_contract_id": "CN202609",
        "contract_frozen_at": "2026-09-07T14:55:01+08:00",
        "quotes": [
            {
                "contract_id": "CN202609",
                "exchange_timestamp": "2026-09-07T15:00:06+08:00",
                "received_at": "2026-09-07T15:00:06.500000+08:00",
                "bid": 99.9,
                "ask": 100.1,
                "last": None,
            },
            {
                "contract_id": "CN202609",
                "exchange_timestamp": "2026-09-07T15:00:12+08:00",
                "received_at": "2026-09-07T15:00:12.500000+08:00",
                "bid": 100.0,
                "ask": 100.2,
                "last": None,
            },
            {
                "contract_id": "CN202609",
                "exchange_timestamp": "2026-09-07T15:03:21+08:00",
                "received_at": "2026-09-07T15:03:21.500000+08:00",
                "bid": 100.4,
                "ask": 100.6,
                "last": None,
            },
            {
                "contract_id": "CN202609",
                "exchange_timestamp": "2026-09-07T15:03:29+08:00",
                "received_at": "2026-09-07T15:03:29.500000+08:00",
                "bid": 100.5,
                "ask": 100.7,
                "last": None,
            },
            {
                "contract_id": "CN202609",
                "exchange_timestamp": "2026-09-07T15:02:00+08:00",
                "received_at": "2026-09-07T15:02:00.200000+08:00",
                "bid": 999.0,
                "ask": 999.2,
                "last": None,
            },
        ],
        "source_entitlement_id": "BROKER_REALTIME_A50_ENTITLEMENT_RECEIPT_V1",
        "source_entitlement_evidence_sha256": "a" * 64,
        "sse_rule_state": "VERIFIED_CURRENT",
        "sgx_session_state": "VERIFIED_CONTINUOUS_TRADING",
        "etf_state": "NORMAL_TRADING",
        "company_action_state": "VERIFIED_NONE_OR_TOTAL_RETURN_ADJUSTED",
        "etf_close": 4.128,
        "etf_close_source_id": "SSE_510300_OFFICIAL_CLOSE_20260907",
        "etf_close_source_sha256": "b" * 64,
        "signal_calculated_at": "2026-09-07T15:03:31+08:00",
    }


def sample_target_payload() -> dict:
    return {
        "signal_trade_date": "2026-09-07",
        "exit_trade_date": "2026-09-08",
        "window_first_exchange_timestamp": "2026-09-08T09:35:00+08:00",
        "window_last_exchange_timestamp": "2026-09-08T09:35:30+08:00",
        "received_at": "2026-09-08T09:35:31+08:00",
        "matured_at": "2026-09-08T09:35:31.100000+08:00",
        "exit_window_bid_vwap": 4.140,
        "exit_window_ask_vwap": 4.142,
        "exit_window_mid": 4.141,
        "exit_quote_depth_state": "VERIFIED_FOR_FROZEN_QUANTITY",
        "company_action_state": "VERIFIED_NONE_OR_TOTAL_RETURN_ADJUSTED",
        "target_source_id": "BROKER_510300_L1_DEPTH_CAPTURE_20260908",
    }


def test_protocol_is_strict_forward_only_and_zero_authority(config: dict) -> None:
    assert config["protocol"]["research_mode"] == "STRICT_FORWARD_ONLY"
    assert config["protocol"]["historical_backfill_allowed"] is False
    assert config["protocol"]["historical_tradable_backtest_allowed"] is False
    assert config["scope"]["tradable_universe"] == ["510300.SH", "CASH_CNY"]
    assert config["scope"]["signal_instrument_position"] == 0
    assert config["protocol"]["position_impact"] == 0
    assert not any(config["boundaries"].values())


def test_source_receipt_separates_verified_rule_from_unverified_sgx_and_broker() -> None:
    receipt = load_json_object(SOURCE_RECEIPT_PATH)
    assert receipt["sse_rule"]["status"] == "PASS_OFFICIAL_RULE_CURRENT"
    assert receipt["sse_rule"]["effective_date"] == "2026-07-06"
    assert receipt["sse_rule"]["current_status_on_rule_page"] == "现行有效"
    assert receipt["sgx_a50_contract_spec"]["admission"] is False
    assert receipt["broker_and_feed"]["admission"] is False
    assert receipt["local_clock"]["admission"] is False


def test_no_backfill_after_forward_start(config: dict) -> None:
    with pytest.raises(ProtocolError, match="禁止补跑或回填"):
        assert_authoritative_trade_date(
            date(2026, 9, 7),
            config,
            datetime.fromisoformat("2026-09-08T15:03:35+08:00"),
        )
    with pytest.raises(ProtocolError, match="不得早于"):
        assert_authoritative_trade_date(
            date(2026, 9, 4),
            config,
            datetime.fromisoformat("2026-09-04T15:03:35+08:00"),
        )


def test_contract_is_frozen_after_1455_from_two_nearest_months(config: dict) -> None:
    payload = sample_daily_payload()
    assert select_frozen_contract(payload, config, date(2026, 9, 7)) == "CN202609"
    assert payload["contract_candidates"][2]["cumulative_volume_0900_1455"] > payload["contract_candidates"][0]["cumulative_volume_0900_1455"]


def test_no_continuous_contract_substitution(config: dict) -> None:
    payload = sample_daily_payload()
    payload["contract_candidates"][0]["is_continuous"] = True
    with pytest.raises(NoViewError) as captured:
        select_frozen_contract(payload, config, date(2026, 9, 7))
    assert captured.value.code == "NO_VIEW_CONTINUOUS_CONTRACT_FORBIDDEN"


def test_contract_volume_tie_is_no_view(config: dict) -> None:
    payload = sample_daily_payload()
    payload["contract_candidates"][1]["cumulative_volume_0900_1455"] = 12000
    with pytest.raises(NoViewError) as captured:
        select_frozen_contract(payload, config, date(2026, 9, 7))
    assert captured.value.code == "NO_VIEW_CONTRACT_VOLUME_TIE"


def test_exchange_timestamp_must_not_follow_received_at(config: dict) -> None:
    payload = sample_daily_payload()
    payload["quotes"][0]["received_at"] = "2026-09-07T15:00:05.900000+08:00"
    with pytest.raises(NoViewError) as captured:
        build_daily_observation(payload, "c" * 64, config)
    assert captured.value.code == "NO_VIEW_RECEIVED_BEFORE_EXCHANGE_TIMESTAMP"


def test_quote_staleness_gate_is_three_seconds(config: dict) -> None:
    payload = sample_daily_payload()
    payload["quotes"][0]["received_at"] = "2026-09-07T15:00:09.100000+08:00"
    with pytest.raises(NoViewError) as captured:
        build_daily_observation(payload, "c" * 64, config)
    assert captured.value.code == "NO_VIEW_QUOTE_STALENESS_EXCEEDED"


def test_signal_quotes_must_arrive_before_150335(config: dict) -> None:
    payload = sample_daily_payload()
    payload["signal_calculated_at"] = "2026-09-07T15:03:35.001000+08:00"
    with pytest.raises(NoViewError) as captured:
        build_daily_observation(payload, "c" * 64, config)
    assert captured.value.code == "NO_VIEW_SIGNAL_DEADLINE_FAILED"


def test_x_uses_only_frozen_windows_and_midpoint_priority(config: dict) -> None:
    row = build_daily_observation(sample_daily_payload(), "c" * 64, config)
    assert row["a50_start_price"] == pytest.approx(100.05)
    assert row["a50_end_price"] == pytest.approx(100.55)
    assert row["x_log_return"] == pytest.approx(math.log(100.55 / 100.05))
    assert row["a50_start_price_method"] == "MEDIAN_FRESH_BID_ASK_MIDPOINT"
    assert row["mu_hat"] is None
    assert row["research_tier_state"] == "NOT_EVALUATED_BEFORE_G2"


def test_trade_median_fallback_requires_three_eligible_trades(config: dict) -> None:
    payload = sample_daily_payload()
    payload["quotes"] = [
        {
            "contract_id": "CN202609",
            "exchange_timestamp": f"2026-09-07T15:00:{second:02d}+08:00",
            "received_at": f"2026-09-07T15:00:{second:02d}.500000+08:00",
            "last": price,
        }
        for second, price in ((6, 100.0), (9, 100.2), (12, 100.1))
    ] + [
        {
            "contract_id": "CN202609",
            "exchange_timestamp": f"2026-09-07T15:03:{second:02d}+08:00",
            "received_at": f"2026-09-07T15:03:{second:02d}.500000+08:00",
            "last": price,
        }
        for second, price in ((21, 100.4), (24, 100.6), (29, 100.5))
    ]
    row = build_daily_observation(payload, "c" * 64, config)
    assert row["a50_start_price"] == pytest.approx(100.1)
    assert row["a50_end_price"] == pytest.approx(100.5)
    assert row["a50_start_price_method"] == "MEDIAN_ELIGIBLE_TRADES"


def test_current_day_and_immature_target_are_excluded_from_model() -> None:
    observations = [
        {"trade_date": "2026-09-07", "observation_state": "ELIGIBLE_RAW_FORWARD_OBSERVATION_PENDING_TARGET", "x_log_return": 0.001},
        {"trade_date": "2026-09-08", "observation_state": "ELIGIBLE_RAW_FORWARD_OBSERVATION_PENDING_TARGET", "x_log_return": 0.002},
        {"trade_date": "2026-09-09", "observation_state": "NO_VIEW", "x_log_return": 0.003},
    ]
    targets = [
        {"signal_trade_date": "2026-09-07", "exit_trade_date": "2026-09-08", "target_state": "MATURE_ELIGIBLE", "y_mid_log_return": 0.0015},
        {"signal_trade_date": "2026-09-08", "exit_trade_date": "2026-10-01", "target_state": "MATURE_ELIGIBLE", "y_mid_log_return": 0.0025},
        {"signal_trade_date": "2026-09-09", "exit_trade_date": "2026-09-10", "target_state": "MATURE_ELIGIBLE", "y_mid_log_return": 0.0035},
    ]
    x, y, dates = select_prior_mature_model_rows(observations, targets, date(2026, 10, 1))
    assert x == [0.001]
    assert y == [0.0015]
    assert dates == ["2026-09-07"]


def test_model_is_no_view_before_60_mature_days() -> None:
    result = fit_hac_ols([0.001] * 59, [0.002] * 59, minimum_observations=60)
    assert result["status"] == "NO_VIEW_INSUFFICIENT_MATURE_OBSERVATIONS"
    assert result["beta"] is None


def test_hac_model_includes_intercept() -> None:
    x = np.linspace(-0.01, 0.01, 60)
    y = 0.0007 + 0.8 * x + np.sin(np.arange(60)) * 0.00001
    result = fit_hac_ols(x, y, minimum_observations=60)
    assert result["status"] == "MODEL_ESTIMATED_MECHANISM_ONLY"
    assert result["alpha"] == pytest.approx(0.0007, abs=5e-6)
    assert result["beta"] == pytest.approx(0.8, abs=5e-3)
    assert np.asarray(result["covariance"]).shape == (2, 2)


def test_conditional_mean_interval_uses_full_intercept_covariance(config: dict) -> None:
    model = {
        "status": "MODEL_ESTIMATED_MECHANISM_ONLY",
        "alpha": 0.01,
        "beta": 0.5,
        "covariance": [[4.0, 1.0], [1.0, 9.0]],
    }
    result = conditional_mean_interval(model, 2.0, config)
    expected_se = math.sqrt(4.0 + 2 * 2.0 * 1.0 + 4.0 * 9.0)
    assert result["mu_hat"] == pytest.approx(1.01)
    assert result["mu_se"] == pytest.approx(expected_se)
    critical = config["model"]["conditional_interval_critical_value"]
    assert result["lcb_975"] == pytest.approx(1.01 - critical * expected_se)
    assert result["ucb_975"] == pytest.approx(1.01 + critical * expected_se)


def test_two_tail_bounds_are_frozen_at_975(config: dict) -> None:
    assert config["model"]["conditional_interval_one_sided_confidence"] == pytest.approx(0.975)
    assert config["model"]["conditional_interval_critical_value"] == pytest.approx(1.959963984540054)


def test_cost_floor_at_25000_notional(config: dict) -> None:
    assert round_trip_cost_bps(25000.0, config, scenario="BASE") == pytest.approx(14.0)
    assert round_trip_cost_bps(25000.0, config, scenario="STRESS") == pytest.approx(24.0)
    assert round_trip_cost_bps(10000.0, config, scenario="BASE") == pytest.approx(20.0)


def test_research_42bp_and_execution_72bp_are_separate(config: dict) -> None:
    research_only = classify_signal(0.001, {"lcb_975": 0.0050, "ucb_975": 0.0055}, config)
    assert research_only["research_tier_state"] == "RESEARCH_BUY"
    assert research_only["execution_tier_state"] == "NO_EXECUTION_SIGNAL"
    execution = classify_signal(-0.001, {"lcb_975": -0.0080, "ucb_975": -0.0075}, config)
    assert execution["research_tier_state"] == "RESEARCH_SELL"
    assert execution["execution_tier_state"] == "EXECUTION_SELL_REQUIRES_SEPARATE_AUTHORIZATION"


def test_research_intent_is_before_1505_but_not_an_order(config: dict) -> None:
    assert config["clock"]["research_intent_at"] == "15:03:45"
    assert config["clock"]["after_hours_matching_window"][0] == "15:05:00"
    assert config["boundaries"]["executable_order_generation_enabled"] is False


def test_theoretical_fill_is_forbidden(config: dict) -> None:
    with pytest.raises(ProtocolError, match="NO_VIEW_NO_LIVE_AUTHORIZATION"):
        assert_actual_fill_recording_authorized(config)


def test_unfilled_counts_as_zero() -> None:
    assert intended_capital_return(0.0, None) == 0.0


def test_partial_fill_scales_intended_capital_return() -> None:
    assert intended_capital_return(0.4, 0.01) == pytest.approx(0.004)
    with pytest.raises(ProtocolError):
        intended_capital_return(0.4, None)


def test_t1_base_lot_rotation_uses_only_prior_day_inventory() -> None:
    lots = [
        {"lot_id": "A", "acquired_date": "2026-09-07", "available_shares": 6000},
        {"lot_id": "B", "acquired_date": "2026-09-08", "available_shares": 6000},
        {"lot_id": "C", "acquired_date": "2026-09-09", "available_shares": 6000},
    ]
    assert select_t1_sell_lot(lots, date(2026, 9, 9)) == "A"
    with pytest.raises(ProtocolError, match="NO_EXECUTABLE_CAPACITY"):
        select_t1_sell_lot([lots[2]], date(2026, 9, 9))


def test_target_maturity_must_be_after_next_0935_window(config: dict) -> None:
    observation = build_daily_observation(sample_daily_payload(), "c" * 64, config)
    payload = sample_target_payload()
    payload["matured_at"] = "2026-09-08T09:35:29+08:00"
    with pytest.raises(NoViewError) as captured:
        build_target_maturity(observation, payload, "d" * 64, date(2026, 9, 8), config)
    assert captured.value.code == "NO_VIEW_TARGET_MATURED_TOO_EARLY"


def test_valid_target_keeps_mid_and_executable_sides_separate(config: dict) -> None:
    observation = build_daily_observation(sample_daily_payload(), "c" * 64, config)
    result = build_target_maturity(observation, sample_target_payload(), "d" * 64, date(2026, 9, 8), config)
    assert result["target_state"] == "MATURE_ELIGIBLE"
    assert result["y_mid_log_return"] == pytest.approx(math.log(4.141 / 4.128))
    assert result["buy_executable_gross_log_return"] == pytest.approx(math.log(4.140 / 4.128))
    assert result["sell_executable_gross_log_return"] == pytest.approx(math.log(4.128 / 4.142))


def test_company_action_unproven_produces_no_view(config: dict) -> None:
    observation = build_daily_observation(sample_daily_payload(), "c" * 64, config)
    payload = sample_target_payload()
    payload["company_action_state"] = "UNPROVEN"
    result = build_target_maturity(observation, payload, "d" * 64, date(2026, 9, 8), config)
    assert result["target_state"] == "NO_VIEW_COMPANY_ACTION_OR_TOTAL_RETURN_UNPROVEN"
    assert result["y_mid_log_return"] is None


def test_monthly_model_vintage_is_immutable(tmp_path: Path, config: dict) -> None:
    local_config = copy.deepcopy(config)
    local_data_root = tmp_path / "data"
    local_data_root.mkdir()
    local_config["storage"]["approved_resolved_data_root"] = str(local_data_root.resolve())
    model = {"status": "MODEL_ESTIMATED_MECHANISM_ONLY", "alpha": 0.0, "beta": 1.0, "covariance": [[1.0, 0.0], [0.0, 1.0]]}
    path = write_model_vintage(tmp_path, local_config, "2026-10", model)
    assert json.loads(path.read_text(encoding="utf-8"))["model_month"] == "2026-10"
    with pytest.raises(ProtocolError, match="禁止覆盖"):
        write_model_vintage(tmp_path, local_config, "2026-10", model)


def test_g0_missing_external_evidence_blocks_collection(config: dict) -> None:
    manifest = {"frozen_files": []}
    evidence = {"project_id": PROJECT_ID, "checks": {}}
    result = evaluate_g0(ROOT, config, manifest, evidence)
    assert result["g0_state"] == "BLOCKED_G0_NO_FORWARD_SIGNAL_COLLECTION"
    assert result["all_pass"] is False
    assert "A50_PRIMARY_FEED_REALTIME_ENTITLEMENT_PROVEN" in result["blockers"]
    assert "LOCAL_CLOCK_SYNC_PASS" in result["blockers"]


def test_no_position_before_separate_authorization(config: dict) -> None:
    assert config["future_position_mapping"]["enabled_now"] is False
    assert config["protocol"]["current_position_target"] == "UNSET"
    assert config["protocol"]["live_trading_authorized"] is False


def test_frozen_manifest_matches_if_present(config: dict) -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("冻结清单将在测试通过后生成")
    manifest = load_json_object(MANIFEST_PATH)
    assert manifest["bundle_status"] == BUNDLE_STATUS
    expected = canonical_json_sha256(manifest, excluded_keys=("manifest_sha256",))
    assert manifest["manifest_sha256"] == expected
    validated = validate_manifest(ROOT, config, expected)
    assert validated["project_id"] == PROJECT_ID
